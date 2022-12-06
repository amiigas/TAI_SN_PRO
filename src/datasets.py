import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class IntMappedZeroPaddedDataset(Dataset):
    def __init__(self, x, y, standarize=True, transforms=None, device="cpu"):
        self.samples = x
        self.targets = y
        self.transforms = transforms

        if standarize:
            self.samples = StandardScaler().fit_transform(self.samples)

        self.samples = torch.from_numpy(self.samples).float().to(device)
        self.targets = torch.from_numpy(self.targets).float().reshape(-1, 1).to(device)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]

        if self.transforms:
            sample = self.transforms(sample)

        return sample, target
