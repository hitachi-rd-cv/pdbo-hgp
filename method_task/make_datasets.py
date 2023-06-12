"""
The following codes are modified or copied from parts of files under (https://github.com/omarfoq/FedEM/blob/main/data).
The following codes are licensed under the Apache License 2.0.
The full license text is available at (https://github.com/omarfoq/FedEM/blob/main/LICENSE).
"""

import json
import os
import random
import re

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, EMNIST, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize

from constants import NamesDataset
from fedem.data.shakespeare.generate_data import train_test_split as train_test_split_shakespeare
from fedem.data.shakespeare.preprocess_shakespeare import _split_into_plays, _get_train_test_by_character, _write_data_by_character
from fedem.data.utils import split_dataset_by_labels, pathological_non_iid_split, pachinko_allocation_split, iid_divide
from fedem.datasets import SubEMNIST, SubCIFAR10, CharacterDataset, SubCIFAR100, TabularDataset
from lib_task.dataset import IndicesDataset, Mean, MyMNIST, SyntheticMixtureDataset


def make_datasets(name_dataset, n_nodes, option_dataset, seed, output_dir):
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
            [ToTensor()]
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
                    n_clusters=-1,
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

        transform_of_clusters = []
        for _ in range(n_components):
            transform_of_clusters.append(Compose([ToTensor(), Normalize((np.random.rand(),), (np.random.rand(),))]))

        all_nodes = list(range(n_nodes))
        random.shuffle(all_nodes)
        clusters = iid_divide(all_nodes, n_components)
        client2cluster = dict()  # maps label to its cluster
        for group_idx, idxs_client in enumerate(clusters):
            for idx in idxs_client:
                client2cluster[idx] = group_idx

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

            transform = transform_of_clusters[client2cluster[client_id]]
            datasets_train.append(SubEMNIST(train_indices, data=inputs, targets=targets, transform=transform))
            datasets_valid.append(SubEMNIST(val_indices, data=inputs, targets=targets, transform=transform))
            datasets_test.append(SubEMNIST(test_indices, data=inputs, targets=targets, transform=transform))


    elif name_dataset == NamesDataset.EMNIST_DIGITS:
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
            [ToTensor()]
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

        inputs = torch.cat([d.data for d in dataset.datasets])
        targets = torch.cat([d.targets for d in dataset.datasets])

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
                    n_clusters=-1,
                    alpha=alpha,
                    frac=s_frac,
                    seed=seed,
                )

        if test_tasks_frac > 0:
            train_clients_indices, test_clients_indices = train_test_split(clients_indices, test_size=test_tasks_frac, random_state=seed)
        else:
            train_clients_indices, test_clients_indices = clients_indices, []

        transform_of_clusters = []
        for _ in range(n_components):
            transform_of_clusters.append(Compose([ToTensor(), Normalize((np.random.rand(),), (np.random.rand(),))]))

        all_nodes = list(range(n_nodes))
        random.shuffle(all_nodes)
        clusters = iid_divide(all_nodes, n_components)
        client2cluster = dict()  # maps label to its cluster
        for group_idx, idxs_client in enumerate(clusters):
            for idx in idxs_client:
                client2cluster[idx] = group_idx

        # extract digits of 1 or 7 instances from data and targets tensors
        are_digits = (targets == 0) | (targets == 1) | (targets == 2) | (targets == 3) | (targets == 4) | (targets == 5) | (targets == 6) | (targets == 7) | (targets == 8) | (targets == 9)
        assert are_digits.sum() > 0
        indices_digits = torch.arange(len(targets))[are_digits]

        datasets_train = []
        datasets_valid = []
        datasets_test = []

        mode, clients_indices = 'train', train_clients_indices
        for client_id, indices in enumerate(clients_indices):
            indices = [i for i in indices if i in indices_digits]

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

            transform = transform_of_clusters[client2cluster[client_id]]
            datasets_train.append(SubEMNIST(train_indices, data=inputs, targets=targets, transform=transform))
            datasets_valid.append(SubEMNIST(val_indices, data=inputs, targets=targets, transform=transform))
            datasets_test.append(SubEMNIST(test_indices, data=inputs, targets=targets, transform=transform))



    elif name_dataset == NamesDataset.CIFAR10:
        n_components = option_dataset['n_components']
        alpha = option_dataset['alpha']
        s_frac = option_dataset['s_frac']
        tr_frac = option_dataset['tr_frac']
        val_frac = option_dataset['val_frac']
        n_classes = option_dataset['n_classes']
        n_shards = option_dataset['n_shards']
        pathological_split = option_dataset['pathological_split']
        test_tasks_frac = option_dataset['test_tasks_frac']

        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        dataset = \
            ConcatDataset([
                CIFAR10(root='./data', download=True, train=True, transform=transform),
                CIFAR10(root='./data', download=False, train=False, transform=transform)
            ])

        if pathological_split:
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

        inputs = torch.cat([torch.tensor(d.data) for d in dataset.datasets])
        targets = torch.cat([torch.tensor(d.targets) for d in dataset.datasets])

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

            datasets_train.append(SubCIFAR10(train_indices, data=inputs, targets=targets))
            datasets_valid.append(SubCIFAR10(val_indices, data=inputs, targets=targets))
            datasets_test.append(SubCIFAR10(test_indices, data=inputs, targets=targets))

    elif name_dataset == NamesDataset.CIFAR100:
        N_FINE_LABELS = 100
        N_COARSE_LABELS = 20

        COARSE_LABELS = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])

        pathological_split = option_dataset['pathological_split']
        pachinko_split = option_dataset['pachinko_allocation_split']
        n_shards = option_dataset['n_shards']
        n_components = option_dataset['n_components']
        alpha = option_dataset['alpha']
        beta = option_dataset['beta']
        s_frac = option_dataset['s_frac']
        tr_frac = option_dataset['tr_frac']
        val_frac = option_dataset['val_frac']
        test_tasks_frac = option_dataset['test_tasks_frac']

        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset = \
            ConcatDataset([
                CIFAR100(root='./data', download=True, train=True, transform=transform),
                CIFAR100(root='./data', download=False, train=False, transform=transform)
            ])

        if pathological_split:
            clients_indices = \
                pathological_non_iid_split(
                    dataset=dataset,
                    n_classes=N_FINE_LABELS,
                    n_clients=n_nodes,
                    n_classes_per_client=n_shards,
                    frac=s_frac,
                    seed=seed
                )
        elif pachinko_split:
            clients_indices = \
                pachinko_allocation_split(
                    dataset=dataset,
                    n_clients=n_nodes,
                    coarse_labels=COARSE_LABELS,
                    n_fine_labels=N_FINE_LABELS,
                    n_coarse_labels=N_COARSE_LABELS,
                    alpha=alpha,
                    beta=beta,
                    frac=s_frac,
                    seed=seed
                )
        else:
            clients_indices = \
                split_dataset_by_labels(
                    dataset=dataset,
                    n_classes=N_FINE_LABELS,
                    n_clients=n_nodes,
                    n_clusters=n_components,
                    alpha=alpha,
                    frac=s_frac,
                    seed=seed
                )

        if test_tasks_frac > 0:
            train_clients_indices, test_clients_indices = \
                train_test_split(
                    clients_indices,
                    test_size=test_tasks_frac,
                    random_state=seed
                )
        else:
            train_clients_indices, test_clients_indices = clients_indices, []

        inputs = torch.cat([torch.tensor(d.data) for d in dataset.datasets])
        targets = torch.cat([torch.tensor(d.targets) for d in dataset.datasets])

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

            datasets_train.append(SubCIFAR100(train_indices, data=inputs, targets=targets))
            datasets_valid.append(SubCIFAR100(val_indices, data=inputs, targets=targets))
            datasets_test.append(SubCIFAR100(test_indices, data=inputs, targets=targets))

    elif name_dataset == NamesDataset.SHAKE_SPEARE:
        RAW_DATA_PATH = "external_repos/fedem/data/shakespeare/raw_data/"
        s_frac = option_dataset['s_frac']
        tr_frac = option_dataset['tr_frac']
        val_frac = option_dataset['val_frac']
        train_tasks_frac = option_dataset['train_tasks_frac']
        thres_n_lines = option_dataset['thres_n_lines']

        print('Splitting .txt data between users')
        input_filename = os.path.join(RAW_DATA_PATH, "raw_data.txt")
        with open(input_filename, 'r') as input_file:
            shakespeare_full = input_file.read()
        plays, discarded_lines = _split_into_plays(shakespeare_full)
        print('Discarded %d lines' % len(discarded_lines))
        users_and_plays, all_examples, _ = _get_train_test_by_character(plays, test_fraction=-1.0, thres_n_lines=thres_n_lines)
        with open(os.path.join(output_dir, 'users_and_plays.json'), 'w') as ouf:
            json.dump(users_and_plays, ouf)
        play_and_character_dir = os.path.join(output_dir, 'by_play_and_character')
        _write_data_by_character(all_examples, play_and_character_dir)

        rng = random.Random(seed)
        n_tasks = int(len(os.listdir(play_and_character_dir)) * s_frac)
        print(f"n_tasks={n_tasks}")
        file_names_list = os.listdir(play_and_character_dir)
        rng.shuffle(file_names_list)

        file_names_list = file_names_list[:n_tasks]
        rng.shuffle(file_names_list)

        datasets_train = []
        datasets_valid = []
        datasets_test = []

        for idx, file_name in enumerate(file_names_list):
            if idx < int(train_tasks_frac * n_tasks):
                pass  # mode = "train"
            else:
                continue  # mode = "test"

            text_path = os.path.join(play_and_character_dir, file_name)

            with open(text_path, "r") as f:
                raw_text = f.read()

            raw_text = re.sub(r"   *", r' ', raw_text)

            train_text, test_text = train_test_split_shakespeare(raw_text, tr_frac)

            if val_frac > 0:
                train_text, val_text = train_test_split_shakespeare(train_text, 1. - val_frac)
                val_text = val_text.replace('\n', ' ')

            else:
                val_text = None

            train_text = train_text.replace('\n', ' ')
            test_text = test_text.replace('\n', ' ')

            dataset_train = CharacterDataset(train_text)
            dataset_valid = CharacterDataset(val_text)
            dataset_test = CharacterDataset(test_text)

            if len(dataset_train) > 0 and len(dataset_valid) > 0 and len(dataset_test) > 0:
                datasets_train.append(CharacterDataset(train_text))
                datasets_valid.append(CharacterDataset(val_text))
                datasets_test.append(CharacterDataset(test_text))

        datasets_train = datasets_train[:n_nodes]
        datasets_valid = datasets_valid[:n_nodes]
        datasets_test = datasets_test[:n_nodes]



    elif name_dataset == NamesDataset.SYNTHETIC:
        np.random.seed(seed)
        dataset = SyntheticMixtureDataset(
            n_components=option_dataset['n_components'],
            n_classes=option_dataset['n_classes'],
            n_tasks=n_nodes,
            dim=option_dataset['dimension'],
            noise_level=option_dataset['noise_level'],
            alpha=option_dataset['alpha'],
            uniform_marginal=option_dataset['uniform_marginal'],
            seed=seed,
            box=option_dataset['box'],
            min_num_samples=option_dataset['min_num_samples'],
            max_num_samples=option_dataset['max_num_samples'],
        )

        datasets_train = []
        datasets_valid = []
        datasets_test = []
        for task_id in range(dataset.n_tasks):
            x_train, y_train = dataset.generate_data(task_id, dataset.num_samples[task_id])
            x_valid, y_valid = dataset.generate_data(task_id, dataset.num_samples[task_id])
            x_test, y_test = dataset.generate_data(task_id, option_dataset['n_test'])
            datasets_train.append(TabularDataset(x_train, y_train, n_classes=option_dataset['n_classes']))
            datasets_valid.append(TabularDataset(x_valid, y_valid, n_classes=option_dataset['n_classes']))
            datasets_test.append(TabularDataset(x_test, y_test, n_classes=option_dataset['n_classes']))

    else:
        raise ValueError(name_dataset)
    # # print dataset info
    pt = PrettyTable(['Node', 'Train', 'Valid', 'Test'])
    for i, (d_tr, d_va, d_te) in enumerate(zip(datasets_train, datasets_valid, datasets_test)):
        pt.add_row([i, len(d_tr), len(d_va), len(d_te)])
    print(pt)

    return datasets_train, datasets_valid, datasets_test
