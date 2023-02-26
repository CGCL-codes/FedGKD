import torch
import torch.nn as nn

MAXLOG = 0.1


class Generator(nn.Module):
    def __init__(self, dataset='mnist', device='cpu', embedding=False, latent_layer_idx=-1):
        super(Generator, self).__init__()
        print("Dataset {}".format(dataset))
        self.embedding = embedding
        self.dataset, self.device = dataset, device
        # self.model=model
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS[dataset]
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss = nn.NLLLoss(reduce=False)  # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim)).to(self.device)  # sampling from Gaussian
        if verbose:
            result['eps'] = eps.to(self.device)
        if self.embedding:  # embedded dense vector
            y_input = self.embedding_layer(labels)
        else:  # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class).to(self.device)
            y_input.zero_()
            # labels = labels.view
            y_input.scatter_(1, labels.view(-1, 1), 1)
        z = torch.cat((eps, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std


#
# class Decoder(nn.Module):
#     """
#     Decoder for both unstructured and image datasets.
#     """
#     def __init__(self, dataset='mnist', latent_layer_idx=-1, n_layers=2, units=32):
#         """
#         Class initializer.
#         """
#         #in_features, out_targets, n_layers=2, units=32):
#         super(Decoder, self).__init__()
#         self.cv_configs, self.input_channel, self.n_class, self.scale, self.noise_dim = GENERATORCONFIGS[dataset]
#         self.hidden_dim = self.scale * self.scale * self.cv_configs[0]
#         self.latent_dim = self.cv_configs[0] * 2
#         self.represent_dims = [self.hidden_dim, self.latent_dim]
#         in_features = self.represent_dims[latent_layer_idx]
#         out_targets = self.noise_dim
#
#         # build layer structure
#         layers = [nn.Linear(in_features, units),
#                   nn.ELU(),
#                   nn.BatchNorm1d(units)]
#
#         for _ in range(n_layers):
#             layers.extend([
#                 nn.Linear(units, units),
#                 nn.ELU(),
#                 nn.BatchNorm1d(units)])
#
#         layers.append(nn.Linear(units, out_targets))
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         """
#         Forward propagation.
#         """
#         out = x.view((x.size(0), -1))
#         out = self.layers(out)
#         return out

class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2 = torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2 = torch.split(layer, chunk_size, dim=0)
        lz = torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps = 1 * 1e-5
        diversity_loss = 1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2 = torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2 = torch.split(layer, chunk_size, dim=0)
        lz = torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps = 1 * 1e-5
        diversity_loss = 1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


CONFIGS_ = {
    # input_channel, n_class, hidden_dim, latent_dim
    'cifar10': ([16, 'M', 32, 'M', 'F'], 3, 10, 128, 64),
    'cifar100-c25': ([32, 'M', 64, 'M', 128, 'F'], 3, 25, 128, 128),
    'cifar100-c30': ([32, 'M', 64, 'M', 128, 'F'], 3, 30, 2048, 128),
    'cifar100-c50': ([32, 'M', 64, 'M', 128, 'F'], 3, 50, 2048, 128),

    # 前面的数组(模型), 初始通道, 类别(分类器的输出维度), 隐藏维度(卷积最后的维度), 最后的维度(分类器的输入维度)

    # 'cifar10': (['R'], 3, 10, 256, 256),
    # 'cifar100': (['R'], 3, 100, 256, 256),

    'emnist': ([6, 16, 'F'], 1, 25, 784, 32),
    'mnist': ([6, 16, 'F'], 1, 10, 784, 32),
    'mnist_cnn1': ([6, 'M', 16, 'M', 'F'], 1, 10, 64, 32),
    'mnist_cnn2': ([16, 'M', 32, 'M', 'F'], 1, 10, 128, 32),
    'celeb': ([16, 'M', 32, 'M', 64, 'M', 'F'], 3, 2, 64, 32)
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'cifar10': (512, 64, 3, 10, 64),
    'cifar100': (512, 64, 3, 100, 64),
    "sst": (768, 768, 768, 5, 768),
    "ag_news": (768, 768, 768, 4, 768),
    "imagenet32": (512, 256 , 3, 1000, 256),
    "tiny-imagenet": (2048, 2048 , 3, 200, 2048),
    'celeb': (128, 32, 3, 2, 32),
    'mnist': (256, 32, 1, 10, 32),
    'mnist-cnn0': (256, 32, 1, 10, 64),
    'mnist-cnn1': (128, 32, 1, 10, 32),
    'mnist-cnn2': (64, 32, 1, 10, 32),
    'mnist-cnn3': (64, 32, 1, 10, 16),
    'emnist': (256, 32, 1, 25, 32),
    'emnist-cnn0': (256, 32, 1, 25, 64),
    'emnist-cnn1': (128, 32, 1, 25, 32),
    'emnist-cnn2': (128, 32, 1, 25, 16),
    'emnist-cnn3': (64, 32, 1, 25, 32),
}

RUNCONFIGS = {
        "ag_news":{
            'unique_labels': 4,
            'generative_alpha': 1,  # used to regulate user training
            'generative_beta': 1,
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'weight_decay': 1e-2,
            'num_pretrain_iters': 20,
        },
        "tiny-imagenet":{
            "generative_alpha": 0.1,
            "generative_beta": 0.1
        },

    'sst':
        {
            'unique_labels': 5,
            'generative_alpha': 1,  # used to regulate user training
            'generative_beta': 1,
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'weight_decay': 1e-2,
            'num_pretrain_iters': 20,
        },
    'cifar10':
        {
            'unique_labels': 10,
            'generative_alpha': 0.1,  # used to regulate user training
            'generative_beta': 1,
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'weight_decay': 1e-2,
            'num_pretrain_iters': 20,
        },
    'cifar100':
        {
            'unique_labels': 100,
            'generative_alpha': 1,  # used to regulate user training
            'generative_beta': 1,
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'weight_decay': 1e-2,
            'num_pretrain_iters': 20,
        },
    'emnist':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 25,
            'generative_alpha': 10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'mnist':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,  # diversity loss
            'unique_labels': 10,  # available labels
            'generative_alpha': 10,  # used to regulate user training
            'generative_beta': 10,  # used to regulate user training
            'weight_decay': 1e-2
        },

    'celeb':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 2,
            'generative_alpha': 10,
            'generative_beta': 10,
            'weight_decay': 1e-2
        },

}

generative_model = Generator("cifar10")
print(sum(p.numel() for p in generative_model.parameters()) / 1e6,)