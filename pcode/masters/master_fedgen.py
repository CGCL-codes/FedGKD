from .master import *
from pcode.models.generator import Generator
import torch.nn.functional as F

class MasterFedgen(Master):
    def __init__(self, conf):
        super().__init__(conf)
        self.device = "cuda" if self.conf.graph.on_cuda else "cpu"
        self.generative_model = Generator(self.conf.data, self.device)

        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=1e-4, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)

        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        # send generator
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            generative_model = self.generative_model.state_dict()
            flatten_model = TensorBuffer(list(generative_model.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

        self.conf.logger.log(f"Master send the generator to workers.")
        dist.barrier()

    def receive_extra_info_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local label counts.")

        # init the placeholders to recv the local models from workers.
        label_counts = dict()
        for selected_client_id in selected_client_ids:
            label_count = torch.zeros(self.conf.num_classes)
            label_counts[selected_client_id] = label_count

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=label_counts[client_id], src=world_id
            )
            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"Master received all local label counts.")

        self.label_weights = []
        self.qualified_labels = []

        for label in range(self.conf.num_classes):
            weights = []
            for user in selected_client_ids:
                weights.append(label_counts[user][label])

            self.qualified_labels.append(label)
            # uniform
            self.label_weights.append(np.array(weights) / np.sum(weights))  # obtain p(y)
        self.label_weights = np.array(self.label_weights).reshape((self.conf.num_classes, -1))


    def train_generator(self, flatten_local_models, epochs=10, verbose=True):
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
        n_teacher_iters = 5
        ensemble_eta = 1

        local_models = {}
        for client_idx, flatten_local_model in flatten_local_models.items():
            _arch = self.clientid2arch[client_idx]
            _model = copy.deepcopy(self.client_models[_arch]).to(self.device)
            _model_state_dict = self.client_models[_arch].state_dict()
            flatten_local_model.unpack(_model_state_dict.values())
            _model.load_state_dict(_model_state_dict)
            local_models[client_idx] = _model

        self.generative_model.train()
        self.generative_model = self.generative_model.cuda()
        for _ in range(epochs):
            for _ in range(n_teacher_iters):
                self.generative_optimizer.zero_grad()
                y = np.random.choice(self.qualified_labels, self.conf.batch_size)
                y_input = torch.tensor(y).to(self.device)
                ## feed to generator
                gen_result = self.generative_model(y_input, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps = gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss = self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss = 0
                for user_idx, user in enumerate(local_models.keys()):
                    user_model = local_models[user]
                    user_model.eval()
                    weight = self.label_weights[y][:, user_idx].reshape(-1, 1)

                    _,user_result_given_gen = user_model(gen_output, start_layer_idx=-1)

                    user_output_logp_ = F.log_softmax(user_result_given_gen, dim=1)
                    teacher_loss_ = torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32).to(self.device))
                    teacher_loss += teacher_loss_

                loss = teacher_loss + ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += teacher_loss
                DIVERSITY_LOSS += ensemble_eta * diversity_loss

        info = "Generator: Teacher Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS / (n_teacher_iters * epochs), DIVERSITY_LOSS / (n_teacher_iters * epochs))
        if verbose:
            self.conf.logger.log(info)
        self.generative_lr_scheduler.step()
        self.generative_model = self.generative_model.cpu()

    def aggregate(self, flatten_local_models):
        fedavg_models = self._avg_over_archs(flatten_local_models)

        self.train_generator(flatten_local_models)
        return fedavg_models

