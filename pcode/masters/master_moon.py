from .master import *

class MasterMoon(Master):
    def __init__(self, conf):
        super().__init__(conf)
        self.prev_models_tb = dict()
        for client_id in self.client_ids:
            self.prev_models_tb[client_id] = []

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        # not real communication cost
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            for flatten_model in self.prev_models_tb[selected_client_id]:
                dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                self.conf.logger.log(
                    f"\tMaster send the history model to process_id={worker_rank}."
                )

        dist.barrier()

    def put_local_model_buffer(self, history_model, client_id):
        local_buffer = self.prev_models_tb[client_id]
        buffer_length = len(local_buffer)
        if buffer_length == 0:
            local_buffer.append(history_model)
        else:
            if buffer_length < self.conf.buffer_length - 1:
                local_buffer.append(local_buffer[-1])

            for i in range(buffer_length - 1, 0, -1):
                local_buffer[i] = local_buffer[i - 1]
            local_buffer[0] = history_model

        self.prev_models_tb[client_id] = local_buffer

    def aggregate(self, flatten_local_models):
        for client_id, flatten_model in flatten_local_models.items():
            self.put_local_model_buffer(flatten_model, client_id)

        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        return fedavg_models

    def activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs, to_send_history=False
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)

        activation_msg = torch.zeros((4, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)
        activation_msg[3, :] = torch.Tensor(
            [len(self.prev_models_tb[client_id]) for client_id in selected_client_ids])

        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()