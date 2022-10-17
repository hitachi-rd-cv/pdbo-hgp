import numpy as np
import torch
from PIL import Image
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


