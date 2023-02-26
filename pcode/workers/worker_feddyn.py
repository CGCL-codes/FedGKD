import torch

from .worker import *


class WorkerFedDyn(Worker):
    def __init__(self, conf):
        super().__init__(conf)
        self.global_weight_collector = None
        self.grad_collector = None

    def extra_init(self):
        self.local_grad = copy.deepcopy(self.model)
        self.grad_dict = self.local_grad.state_dict()
        self.grad_tb = TensorBuffer(list(self.grad_dict.values()), use_cuda=False)

    def prepare_train(self):
        self._prepare_train()
        self.local_grad = self.local_grad.to(self.device)
        self.global_weight_collector, self.grad_collector = self.get_params(self.init_model), self.get_params(
            self.local_grad)

    def recv_extra_info_from_master(self):
        old_buffer = copy.deepcopy(self.grad_tb.buffer)
        dist.recv(tensor=self.grad_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.grad_tb.buffer)
        self.grad_tb.buffer = self.grad_tb.buffer.to(self.device)

        self.grad_tb.unpack(self.grad_dict.values(), use_cuda=True)
        self.local_grad.load_state_dict(self.grad_dict)

        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the gradient ({self.arch}) from Master. The gradient status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )

        dist.barrier()

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature):
        local_par_list = self.get_params(self.model)

        loss2 = (self.conf.local_prox_term / 2.0) \
                * (torch.sum(torch.square(local_par_list - self.global_weight_collector)
                             + 2 * self.grad_collector * local_par_list))
        if self.tracker is not None:
            self.tracker.update_local_metrics(
                loss2.item(), -1, n_samples=output.size(0)
            )
        loss = loss + loss2

        return loss
