from .master import *

class MasterFedDyn(Master):
    def __init__(self, conf):
        super().__init__(conf)
        zero_model = copy.deepcopy(self.master_model)
        new_dict = {}
        for key, value in zero_model.state_dict().items():
            new_dict[key] = torch.zeros_like(value)
        zero_model.load_state_dict(new_dict)

        self.local_param_list = [ModuleState(copy.deepcopy(zero_model.state_dict())) for _ in
                                 range(self.conf.n_clients + 1)]

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        # send gradient
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            flatten_model = TensorBuffer(list(self.local_param_list[selected_client_id - 1].state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

        self.conf.logger.log(f"Master send the local grad to workers.")
        dist.barrier()

    def aggregate(self, flatten_local_models):
        # directly averaging.
        weight = float(1.0 / len(flatten_local_models))

        local_states = {}
        for client_idx, flatten_local_model in flatten_local_models.items():
            _arch = self.clientid2arch[client_idx]
            _model_state_dict = copy.deepcopy(self.client_models[_arch].state_dict())
            flatten_local_model.unpack(_model_state_dict.values())
            local_states[client_idx] = ModuleState(_model_state_dict)

        avg_model = copy.deepcopy(self.master_model)
        master_state = ModuleState(copy.deepcopy(self.master_model.state_dict()))

        avg_model_state = None

        for client_idx, model_state in local_states.items():
            self.local_param_list[client_idx] += model_state - master_state

            if avg_model_state is None:
                avg_model_state = model_state * weight
            else:
                avg_model_state += (model_state * weight)

        weight = float(1.0 / self.conf.n_clients)

        for local_param in self.local_param_list[1:]:
            avg_model_state += local_param * weight

        avg_model_state.copy_to_module(avg_model)

        return {_arch: avg_model}
