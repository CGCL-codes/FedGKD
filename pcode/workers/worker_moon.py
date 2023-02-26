from .worker import *

class WorkerMoon(Worker):
    def __init__(self, conf):
        super().__init__(conf)

    def prepare_train(self):
        self._prepare_train()
        for i in range(self.models_buffer_len):
            self.models_buffer[i].to(self.device)

    def listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((4, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)

        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:3, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )
        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )

        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))

        self.models_buffer_len = msg[3][self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()

        self.metrics = create_metrics.Metrics(self.model, task="classification")

        prev_model = copy.deepcopy(self.model).cpu()
        self.prev_model = self._turn_off_grad(prev_model)
        self.buffer_model_dicts = [self.prev_model.state_dict() for _ in range(self.models_buffer_len)]
        self.buffer_model_tbs = [TensorBuffer(list(self.buffer_model_dicts[i].values())) for i in
                                 range(self.models_buffer_len)]
        self.models_buffer = [self.prev_model for _ in range(self.models_buffer_len)]

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


    def recv_extra_info_from_master(self):
        for i in range(self.models_buffer_len):
            # old_buffer = copy.deepcopy(self.buffer_model_tbs[i].buffer)
            dist.recv(self.buffer_model_tbs[i].buffer, src=0)
            # new_buffer = copy.deepcopy(self.buffer_model_tbs[i].buffer)
            self.buffer_model_tbs[i].unpack(self.buffer_model_dicts[i].values())
            self.models_buffer[i].load_state_dict(self.buffer_model_dicts[i])
            self.models_buffer[i] = self.models_buffer[i].to(self.device)
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the historical local model ({self.arch}) from Master."
            )

        dist.barrier()

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature):
        if self.conf.distillation_coefficient == 0 or self.conf.graph.comm_round <= 1:
            return loss

        bsz = data_batch["target"].size(0)

        teacher_feature, _ = self.init_model(data_batch["input"])
        logits = self.similarity(feature, teacher_feature.detach()).reshape(-1, 1)

        for i in range(self.models_buffer_len):
            prev_feature, _ = self.models_buffer[i](data_batch["input"])
            nega = self.similarity(feature, prev_feature.detach()).reshape(-1, 1)
            logits = torch.cat((logits, nega), dim=1)

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits /= self.conf.temperature
        labels = torch.zeros(bsz).to(self.device).long()
        loss2 = self.conf.distillation_coefficient * self.criterion(logits, labels)
        loss = loss + loss2

        if self.tracker is not None:
            self.tracker.update_local_metrics(
                loss2.item(), -1, n_samples=bsz
            )

        return loss


    def similarity(self, x1, x2):
        sim = F.cosine_similarity(x1, x2, dim=-1)
        return sim

