from .master import *

class MasterFedEnsemble(Master):
    def __init__(self, conf):
        super().__init__(conf)
        self.ensembled_teacher = copy.deepcopy(self.master_model)
        self.ensemble_coordinator = create_coordinator.Coordinator(conf, self.metrics)
        self.models_buffer = []

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


