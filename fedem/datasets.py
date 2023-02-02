# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of FedEM (https://github.com/omarfoq/FedEM).
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/omarfoq/FedEM/blob/main/LICENSE).
# ------------------------------------------------------------------------
import string

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from fedem.utils.constants import SHAKESPEARE_CONFIG


class SubEMNIST(Dataset):
    def __init__(self, indices, data, targets, classes_use=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param data: EMNIST dataset inputs
        :param targets: EMNIST dataset labels
        :param transform:
        """
        # assert len(indices) > 0
        self.indices = indices
        self.classes = EMNIST._all_classes
        self.classes_use = classes_use

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])
        else:
            self.transform = transform

        self.data, self.targets = data, targets

        if self.classes_use is not None:
            self.indices = torch.tensor([idx for idx, t in enumerate(self.targets) if (t.item() in self.classes_use) and (idx in self.indices)])
            self.classes = [self.classes[i] for i in classes_use]

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

        if self.classes_use is not None:
            map_idx = {old_id: new_id for new_id, old_id in enumerate(self.classes_use)}
            self.targets = torch.tensor([map_idx[t.item()] for t in self.targets])

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, indices, data, targets, classes_use=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param data: Cifar-10 dataset inputs stored as torch.tensor
        :param targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """
        # assert len(indices) > 0
        self.indices = indices
        self.classes = [str(x) for x in range(10)]
        self.classes_use = classes_use

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        self.data, self.targets = data, targets

        if self.classes_use is not None:
            self.indices = torch.tensor([idx for idx, t in enumerate(self.targets) if (t.item() in self.classes_use) and (idx in self.indices)])
            self.classes = [self.classes[i] for i in classes_use]

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

        if self.classes_use is not None:
            map_idx = {old_id: new_id for new_id, old_id in enumerate(self.classes_use)}
            self.targets = torch.tensor([map_idx[t.item()] for t in self.targets])

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, indices, data, targets, classes_use=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:
        """
        # assert len(indices) > 0
        self.indices = indices
        self.classes = [str(x) for x in range(100)]
        self.classes_use = classes_use

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        self.data, self.targets = data, targets

        if self.classes_use is not None:
            self.indices = torch.tensor([idx for idx, t in enumerate(self.targets) if (t.item() in self.classes_use) and (idx in self.indices)])
            self.classes = [self.classes[i] for i in classes_use]

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

        if self.classes_use is not None:
            map_idx = {old_id: new_id for new_id, old_id in enumerate(self.classes_use)}
            self.targets = torch.tensor([map_idx[t.item()] for t in self.targets])

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class CharacterDataset(Dataset):
    def __init__(self, text):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.classes = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = SHAKESPEARE_CONFIG["chunk_len"]

        self.text = text

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx + self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx + 1:idx + self.chunk_len + 1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx
