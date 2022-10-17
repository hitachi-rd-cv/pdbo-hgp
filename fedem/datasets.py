# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of FedEM (https://github.com/omarfoq/FedEM), Copyright {copyright_marfoq}.
# The portions of the following codes are licensed under the {license_type_marfoq}.
# The full license text is available at ({license_url_marfoq}).
# ------------------------------------------------------------------------
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize


class SubEMNIST(Dataset):
    def __init__(self, indices, emnist_data, emnist_targets, classes_use=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
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

        self.data, self.targets = emnist_data, emnist_targets

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
