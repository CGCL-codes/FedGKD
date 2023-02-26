from .worker import *
from ..models.generator import *


class WorkerFedGen(Worker):
    def __init__(self, conf):
        super().__init__(conf)
        self.generative_model = Generator(self.conf.data, self.device)
        self.generator_state_dict = self.generative_model.state_dict()
        self.generator_tb = TensorBuffer(list(self.generator_state_dict.values()))

    def extra_init(self):
        self.generative_model.cpu()
    def prepare_train(self):
        self._prepare_train()
        self.clean_up_counts()
        self.generative_model.to(self.device)

    def recv_extra_info_from_master(self):
        dist.recv(tensor=self.generator_tb.buffer, src=0)
        self.generator_tb.unpack(self.generator_state_dict.values())
        self.generative_model.load_state_dict(self.generator_state_dict)

        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the generator ({self.arch}) from Master."
        )
        dist.barrier()

    def send_extra_info_to_master(self):
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the label_counts back to Master."
        )
        label_counts = self.label_counts.detach().cpu()
        dist.send(tensor=label_counts, dst=0)
        dist.barrier()

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature):
        if self.conf.graph.comm_round <= 1:
            return loss
        original_generative_alpha = RUNCONFIGS[self.conf.data]['generative_alpha']

        self.generative_model.eval()
        generative_alpha = self.exp_lr_scheduler(self.conf.graph.comm_round, decay=0.98, init_lr=original_generative_alpha)

        sampled_y = np.random.choice(self.conf.num_classes, self.conf.batch_size)
        sampled_y = torch.tensor(sampled_y).to(self.device)
        gen_result = self.generative_model(sampled_y, latent_layer_idx=-1)
        gen_output = gen_result['output']  # latent representation when latent = True, x otherwise

        _,user_output = self.model(gen_output, start_layer_idx=-1)

        loss2 = generative_alpha * torch.mean(
            self.criterion(user_output, sampled_y)
        )

        if self.tracker is not None:
            self.tracker.update_local_metrics(
                loss2.item(), -1, n_samples=output.size(0)
            )
        loss = loss + loss2

        return loss

    def clean_up_counts(self):
        self.label_counts = torch.ones(self.conf.num_classes)

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)-1] += count

    def get_count_labels(self, y):
        result = {}
        unique_y, counts = torch.unique(y, return_counts=True)
        unique_y = unique_y.detach().cpu().numpy()
        counts = counts.detach().cpu().numpy()
        result['labels'] = unique_y
        result['counts'] = counts
        return result