from .worker import *


class WorkerFedProx(Worker):
    def __init__(self, conf):
        super().__init__(conf)
        self.global_weight_collector = None

    def prepare_train(self):
        self._prepare_train()
        self.global_weight_collector = self.get_params(self.init_model)

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature):
        assert self.global_weight_collector is not None

        local_par_list = self.get_params(self.model)

        loss2 = (self.conf.local_prox_term/2.0) * torch.sum(torch.square(local_par_list-self.global_weight_collector))

        if self.tracker is not None:
            self.tracker.update_local_metrics(
                loss2.item(), -1, n_samples=output.size(0)
            )
        loss = loss + loss2

        return loss
