import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST, EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from constants import NamesDataset
from fedem.data.emnist.utils import split_dataset_by_labels, pathological_non_iid_split
from fedem.datasets import SubEMNIST
from lib_task.dataset import IndicesDataset, Mean, MyMNIST


def make_datasets(name_dataset, n_nodes, option_dataset, seed):
    if name_dataset == NamesDataset.TOY_MNIST:
        n_samples_train = option_dataset['n_samples_train']
        n_samples_valid = option_dataset['n_samples_valid']

        indice_train_ori = IndicesDataset(60000)
        indice_test_ori = IndicesDataset(10000)

        n_remainders = len(indice_train_ori) - (n_samples_train + n_samples_valid)
        indice_train, indice_valid, _ = random_split(indice_train_ori, list([n_samples_train, n_samples_valid, n_remainders]))

        # distribute train to n_nodes
        n_samples_train_nodes = [n_samples_train // n_nodes] * n_nodes
        n_samples_train_wasted = [len(indice_train) % n_nodes]
        indices_train = random_split(indice_train, n_samples_train_nodes + n_samples_train_wasted)[:-1]

        # distribute valid to n_nodes
        n_samples_valid_nodes = [n_samples_valid // n_nodes] * n_nodes
        n_samples_valid_wasted = [len(indice_valid) % n_nodes]
        indices_valid = random_split(indice_valid, n_samples_valid_nodes + n_samples_valid_wasted)[:-1]

        # distribute test to n_nodes
        n_samples_test_nodes = [len(indice_test_ori) // n_nodes] * n_nodes
        n_samples_test_wasted = [len(indice_test_ori) % n_nodes]
        indices_test = random_split(indice_test_ori, n_samples_test_nodes + n_samples_test_wasted)[:-1]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            Mean(),
        ])

        n_classes = option_dataset['n_classes']
        classes = np.random.permutation(np.arange(len(MNIST.classes)))[:n_classes]
        datasets_train = [MyMNIST(dataset[:], classes_use=classes, root='./data', train=True, download=True, transform=transform) for dataset in indices_train]
        datasets_valid = [MyMNIST(dataset[:], classes_use=classes, root='./data', train=True, download=True, transform=transform) for dataset in indices_valid]
        datasets_test = [MyMNIST(dataset[:], classes_use=classes, root='./data', train=False, download=True, transform=transform) for dataset in indices_test]

    elif name_dataset == NamesDataset.EMNIST:
        n_components = option_dataset['n_components']
        alpha = option_dataset['alpha']
        s_frac = option_dataset['s_frac']
        tr_frac = option_dataset['tr_frac']
        val_frac = option_dataset['val_frac']
        n_classes = option_dataset['n_classes']
        n_shards = option_dataset['n_shards']
        pathological_split = option_dataset['pathological_split']
        test_tasks_frac = option_dataset['test_tasks_frac']

        transform = Compose(
            [ToTensor(),
             Normalize((0.1307,), (0.3081,))
             ]
        )

        dataset = ConcatDataset([
            EMNIST(
                root='./data',
                split="byclass",
                download=True,
                train=True,
                transform=transform,
            ),
            EMNIST(root='./data',
                   split="byclass",
                   download=False,
                   train=False,
                   transform=transform)
        ])

        if pathological_split:
            assert NotImplementedError
            clients_indices = \
                pathological_non_iid_split(
                    dataset=dataset,
                    n_classes=n_classes,
                    n_clients=n_nodes,
                    n_classes_per_client=n_shards,
                    frac=s_frac,
                    seed=seed,
                )
        else:
            clients_indices = \
                split_dataset_by_labels(
                    dataset=dataset,
                    n_classes=n_classes,
                    n_clients=n_nodes,
                    n_clusters=n_components,
                    alpha=alpha,
                    frac=s_frac,
                    seed=seed,
                )

        if test_tasks_frac > 0:
            train_clients_indices, test_clients_indices = train_test_split(clients_indices, test_size=test_tasks_frac, random_state=seed)
        else:
            train_clients_indices, test_clients_indices = clients_indices, []

        inputs = torch.cat([d.data for d in dataset.datasets])
        targets = torch.cat([d.targets for d in dataset.datasets])

        datasets_train = []
        datasets_valid = []
        datasets_test = []

        mode, clients_indices = 'train', train_clients_indices
        for client_id, indices in enumerate(clients_indices):
            train_indices, test_indices = \
                train_test_split(
                    indices,
                    train_size=tr_frac,
                    random_state=seed,
                )

            if val_frac > 0:
                train_indices, val_indices = \
                    train_test_split(
                        train_indices,
                        train_size=1. - val_frac,
                        random_state=seed,
                    )
            else:
                val_indices = []

            datasets_train.append(SubEMNIST(train_indices, emnist_data=inputs, emnist_targets=targets))
            datasets_valid.append(SubEMNIST(val_indices, emnist_data=inputs, emnist_targets=targets))
            datasets_test.append(SubEMNIST(test_indices, emnist_data=inputs, emnist_targets=targets))

    # print dataset info
    n_classes = len(datasets_train[0].classes)
    pt = PrettyTable(['Node', 'Train', 'Valid', 'Test', *[str(x) for x in range(n_classes)]])
    for i, (d_tr, d_va, d_te) in enumerate(zip(datasets_train, datasets_valid, datasets_test)):
        pt.add_row([i, len(d_tr), len(d_va), len(d_te), *[f"({torch.sum(d_tr.targets == x).item()}, {torch.sum(d_va.targets == x).item()}, {torch.sum(d_te.targets == x).item()})" for x in range(n_classes)]])
    print(pt)

    return datasets_train, datasets_valid, datasets_test
