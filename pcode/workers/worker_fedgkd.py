from .worker import *


class WorkerFedGKD(Worker):
    def __init__(self, conf):
        super().__init__(conf)

        self.model_weights = None
        self.models_buffer = []
        self.avg_teacher = None
        self.init_model_valid_loss = 0

    def prepare_train(self):
        self._prepare_train()
        if self.avg_teacher is None:
            self.avg_teacher = self.init_model

    def recv_extra_info_from_master(self):
        if self.conf.buffer_length <= 1:
            return

        model_tb = copy.deepcopy(self.model_tb)
        self.avg_teacher = copy.deepcopy(self.model)
        dist.recv(model_tb.buffer, src=0)
        model_tb.unpack(self.model_state_dict.values())

        self.avg_teacher.load_state_dict(self.model_state_dict)
        self.avg_teacher = self._turn_off_grad(self.avg_teacher.to(self.device))
        self.conf.logger.log(
            f"The client-{self.conf.graph.client_id}) receive the teacher model."
        )

        dist.barrier()

    def listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((4, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)

        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:3, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )

        if not self.conf.avg_param:  # FedGKD-VOTE
            self.init_model_valid_loss = msg[3][self.conf.graph.rank - 1].to(float).cpu().numpy().tolist()
            self.put_global_model_buffer({'valid_loss': self.init_model_valid_loss, 'model': self.init_model},
                                         len(self.models_buffer))
            self.get_global_weights()

            self.conf.logger.log(
                f"The model buffer length of worker-{self.conf.graph.worker_id}(client-{self.conf.graph.client_id}) = {length}."
            )

        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )

        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))

        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

        self.train_loader, _ = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            # localdata_id start from 0 to the # of clients - 1.
            # client_id starts from 1 to the # of clients.
            localdata_id=self.conf.graph.client_id - 1,
            is_train=True,
            data_partitioner=self.data_partitioner,
        )

    def distillation_with_multiple(self, output, input):
        with torch.no_grad():
            loss2 = 0
            for i, teacher_model in enumerate(self.models_buffer):
                _, teacher_logits = teacher_model['model'](input)

                loss2 += self._divergence(
                    student_logits=output / self.conf.temperature,
                    teacher_logits=teacher_logits / self.conf.temperature,
                ) * self.model_weights[i]

        return loss2

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature):
        if self.conf.distillation_coefficient == 0 or self.conf.graph.comm_round <= 1:
            return loss

        loss2 = 0
        input = data_batch["input"]
        student_logits = output

        if self.avg_teacher is None:  # FedGKD-VOTE
            loss2 += self.distillation_with_multiple(student_logits, input)
        else:  # FedGKD
            with torch.no_grad():
                _, teacher_logits = self.avg_teacher(input)

            if self.conf.loss_type == 'mse':
                loss2 += F.mse_loss(student_logits, teacher_logits)
            elif self.conf.loss_type == 'kl':
                loss2 += self._divergence(
                    student_logits=student_logits / self.conf.temperature,
                    teacher_logits=teacher_logits / self.conf.temperature,
                )
            else:
                assert "not support loss type!"

        loss2 = self.conf.distillation_coefficient * loss2
        loss = loss + loss2

        if self.tracker is not None and loss2 != 0:
            self.tracker.update_local_metrics(
                loss2.item(), -1, n_samples=data_batch["target"].size(0)
            )
        return loss

    def mse_divergence(self, student_logits, teacher_logits):
        divergence = F.mse_loss(
            student_logits, teacher_logits
        )
        return divergence

    def _divergence(self, student_logits, teacher_logits):
        divergence = self.conf.temperature * self.conf.temperature * F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def put_global_model_buffer(self, history_model, length):
        if length >= self.conf.buffer_length:
            self.models_buffer.pop(0)
        self.models_buffer.append(copy.deepcopy(history_model))

    def get_global_weights(self):

        weights = []
        for model_dict in self.models_buffer:
            weights.append(-model_dict['valid_loss'])

        weights = torch.FloatTensor(weights)
        self.model_weights = torch.softmax(weights / self.conf.loss_temperature, dim=0)
        self.model_weights.requires_grad = False
