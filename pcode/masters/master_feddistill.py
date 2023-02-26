from .master import *

class MasterFedDistill(Master):
    def __init__(self, conf):
        super().__init__(conf)

        self.global_logits = torch.zeros(self.conf.num_classes, self.conf.num_classes)
        self.label_counts = [torch.ones(self.conf.num_classes) for _ in range(self.conf.n_clients)]
        self.logit_sums = [torch.zeros(self.conf.num_classes, self.conf.num_classes) for _ in
                           range(self.conf.n_clients)]

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        # send logits
        self.global_logits = torch.zeros(self.conf.num_classes, self.conf.num_classes)
        logit_avg_map = dict()
        for client_id in self.client_ids:
            logit_avg = self.logit_sums[client_id - 1] / self.label_counts[client_id - 1].float().unsqueeze(1)
            logit_avg_map[client_id] = logit_avg
            self.global_logits += logit_avg

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            logit_send = (self.global_logits - logit_avg_map[selected_client_id]) / (self.conf.n_clients - 1)

            dist.send(tensor=logit_send.clone().detach().cpu(), dst=worker_rank)
            dist.send(tensor=self.logit_sums[selected_client_id - 1].clone().detach().cpu(), dst=worker_rank)
            dist.send(tensor=self.label_counts[selected_client_id - 1].clone().detach().cpu(), dst=worker_rank)

        self.conf.logger.log(f"Master send the logits to clients")
        dist.barrier()

    def receive_extra_info_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local logits.")

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=self.logit_sums[client_id - 1], src=world_id
            )
            reqs.append(req)

        for req in reqs:
            req.wait()

        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=self.label_counts[client_id - 1], src=world_id
            )
            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"Master received all local logits.")

