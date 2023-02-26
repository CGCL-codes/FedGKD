from .worker import *
from ..utils.stat_tracker import LogitTracker


class WorkerFedDistill(Worker):
    def __init__(self, conf):
        super().__init__(conf)

        self.global_logits = torch.zeros(self.conf.num_classes, self.conf.num_classes)
        self.logit_tracker = LogitTracker(unique_labels=self.conf.num_classes)

    def prepare_train(self):
        self._prepare_train()
        self.global_logits = self.global_logits.to(self.device)
        self.logit_tracker.logit_sums = self.logit_tracker.logit_sums.to(self.device)
        self.logit_tracker.label_counts = self.logit_tracker.label_counts.to(self.device)

    def recv_extra_info_from_master(self):

        dist.recv(tensor=self.global_logits.clone().detach().cpu(), src=0)
        dist.recv(tensor=self.logit_tracker.logit_sums.clone().detach().cpu(), src=0)
        dist.recv(tensor=self.logit_tracker.label_counts.clone().detach().cpu(), src=0)

        dist.barrier()

    def send_extra_info_to_master(self):
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the logits back to Master."
        )

        logit_sums = self.logit_tracker.logit_sums.clone().detach().cpu()
        label_counts = self.logit_tracker.label_counts.clone().detach().cpu()
        dist.send(tensor=logit_sums, dst=0)
        dist.send(tensor=label_counts, dst=0)
        dist.barrier()

    def _divergence(self, student_logits, teacher_logits):
        divergence = self.conf.temperature * self.conf.temperature * F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature):
        self.logit_tracker.update(logits=output, Y=data_batch["target"])

        if self.conf.distillation_coefficient == 0 or self.conf.graph.comm_round <= 1:
            return loss
        loss2 = self.conf.distillation_coefficient * self._divergence(output / self.conf.temperature,
                                 self.global_logits[data_batch["target"],:] / self.conf.temperature)

        loss = loss + loss2
        if self.tracker is not None:
            self.tracker.update_local_metrics(
                loss2.item(), -1, n_samples=output.size(0)
            )

        return loss


