import pickle

import numpy as np
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torchvision import datasets


class IndicesDataset(Dataset):
    def __init__(self, n_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.arange(n_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Mean:
    def __call__(self, img):
        return torch.mean(img, dim=(1, 2))


class MyMNIST(datasets.MNIST):
    def __init__(self, indices, *args, classes_use=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(indices) > 0
        self.indices = indices
        self.classes_use = classes_use

        if self.classes_use is not None:
            self.indices = torch.tensor([idx for idx, t in enumerate(self.targets) if (t.item() in self.classes_use) and (idx in self.indices)])
            self.classes = [self.classes[i] for i in classes_use]

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

        if self.classes_use is not None:
            map_idx = {old_id: new_id for new_id, old_id in enumerate(self.classes_use)}
            self.targets = torch.tensor([map_idx[t.item()] for t in self.targets])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class SyntheticMixtureDataset:
    def __init__(
            self,
            n_classes,
            n_components,
            n_tasks,
            dim,
            noise_level,
            alpha,
            box,
            uniform_marginal,
            min_num_samples,
            max_num_samples,
            seed
    ):

        if n_classes != 2:
            raise NotImplementedError("Only binary classification is supported for the moment")

        self.n_classes = n_classes
        self.n_components = n_components
        self.n_tasks = n_tasks
        self.dim = dim
        self.noise_level = noise_level
        self.alpha = alpha * np.ones(n_components)
        self.box = box
        self.uniform_marginal = uniform_marginal
        self.seed = seed

        np.random.seed(self.seed)
        self.num_samples = get_num_samples(self.n_tasks, min_num_samples, max_num_samples)

        self.theta = np.zeros((self.n_components, self.dim))
        self.mixture_weights = np.zeros((self.n_tasks, self.n_components))

        self.generate_mixture_weights()
        self.generate_components()

    def generate_mixture_weights(self):
        for task_id in range(self.n_tasks):
            self.mixture_weights[task_id] = np.random.dirichlet(alpha=self.alpha)

    def generate_components(self):
        self.theta = np.random.uniform(self.box[0], self.box[1], size=(self.n_components, self.dim))

    def generate_data(self, task_id, n_samples=10_000):
        latent_variable_count = np.random.multinomial(n_samples, self.mixture_weights[task_id])
        y = np.zeros(n_samples)

        if self.uniform_marginal:
            x = np.random.uniform(self.box[0], self.box[1], size=(n_samples, self.dim))
        else:
            raise NotImplementedError("Only uniform marginal is available for the moment")

        current_index = 0
        for component_id in range(self.n_components):
            y_hat = x[current_index:current_index + latent_variable_count[component_id]] @ self.theta[component_id]
            noise = np.random.normal(size=latent_variable_count[component_id], scale=self.noise_level)
            y[current_index: current_index + latent_variable_count[component_id]] = \
                np.round(sigmoid(y_hat + noise)).astype(int)

        return shuffle(x, y)

    def save_metadata(self, path_):
        metadata = dict()
        metadata["mixture_weights"] = self.mixture_weights
        metadata["theta"] = self.theta

        with open(path_, 'wb') as f:
            pickle.dump(metadata, f)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_num_samples(num_tasks, min_num_samples=50, max_num_samples=1000):
    num_samples = np.random.lognormal(4, 2, num_tasks).astype(int)
    num_samples = [min(s + min_num_samples, max_num_samples) for s in num_samples]
    return num_samples
