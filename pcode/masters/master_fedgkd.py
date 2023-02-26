from .master import *

class MasterFedGKD(Master):
    def __init__(self, conf):
        super().__init__(conf)
        self.ensembled_teacher = copy.deepcopy(self.master_model)
        self.ensemble_coordinator = create_coordinator.Coordinator(conf, self.metrics)
        self.models_buffer = []

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        if self.conf.buffer_length <= 1:
            return
        # send teacher
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            teacher_model_state_dict = self.ensembled_teacher.state_dict()
            flatten_model = TensorBuffer(list(teacher_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            self.conf.logger.log(
                f"\tMaster send the ensembled teacher to process_id={worker_rank}."
            )

        dist.barrier()

    def additional_validation(self):
        self.conf.logger.log("Evaluating the ensembled global model.")
        master_utils.do_validation(
            self.conf,
            self.ensemble_coordinator,
            self.ensembled_teacher,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label=f"ensembled_test_loader_",
        )

    def ensemble_historical_models(self, history_model):
        if len(self.models_buffer) >= self.conf.buffer_length:
            self.models_buffer.pop(0)
        self.models_buffer.append(copy.deepcopy(history_model))

        # avg historical models
        models = []
        for model in self.models_buffer:
            models.append(model.state_dict())

        avg_state_dict = {}
        avg_weight = 1.0 / float(len(self.models_buffer))

        for param_name, param in self.master_model.state_dict().items():
            avg_state_dict[param_name] = torch.zeros_like(param.data)
            for model in models:
                avg_state_dict[param_name] = avg_state_dict[param_name] + model[param_name] * avg_weight

        self.ensembled_teacher.load_state_dict(avg_state_dict)

    def aggregate(self, flatten_local_models):
        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        fedavg_model = list(fedavg_models.values())[0]
        self.ensemble_historical_models(fedavg_model)

        return fedavg_models

    def _activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs, to_send_history=False
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        msg_len = 4

        activation_msg = torch.zeros((msg_len, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)

        if not self.conf.avg_param: # FedGKD-VOTE
            valid_performance = master_utils.validate(conf=self.conf,
                                                      coordinator=None,
                                                      model=self.master_model,
                                                      criterion=self.criterion,
                                                      data_loader=self.val_loader,
                                                      label=None,
                                                      display=False,
                                                      metrics=self.metrics)

            activation_msg[3, :] = valid_performance['loss']
            self.conf.logger.log(
                f"comm_round:{self.conf.graph.comm_round}, valiation dataset loss:{valid_performance['loss']:.4f}")

        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()


