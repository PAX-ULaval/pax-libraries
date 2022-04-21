import pickle as pk

import numpy as np
import pandas as pd
import torch
from paxutils.path import Path
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, csv_file, transform=None, columns=None):

        self.data = pd.read_csv(csv_file)

        if columns is None:
            self.columns = ["age", "ethnicity", "gender"]
        else:
            self.columns = columns

        self.data.drop(columns={"img_name"}, inplace=True)
        self.data["pixels"] = self.data["pixels"].apply(
            lambda x: np.array(x.split(), dtype="float32").reshape((1, 48, 48)) / 255
        )
        self.data["age"] = self.data["age"].apply(lambda x: np.array([x], dtype="float32"))
        self.X = torch.Tensor(self.data["pixels"])

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indice):
        if torch.is_tensor(indice):
            indice = indice.tolist()
        image = self.X[indice]
        if len(self.columns) > 1:
            attribute = torch.Tensor([float(self.data.iloc[indice][i]) for i in self.columns])
        else:
            attribute = self.data.iloc[indice][self.columns[0]]
        sample = (image, attribute)

        if self.transform:
            sample = (self.transform(sample[0]), attribute)

        return sample


class EchantillonCIFAR10(Dataset):
    """
        Échantillon du jeu de donnée CIFAR10. L'échantillon comprend 10 000 données d'entraînements
        et 2 000 de tests.
    Args:
        train (bool): Si True, prend les données d'entraînements, sinon les données de tests
        transform (callable, optional): Une function/transform qui prend le target et le transforme.

    """

    def __init__(self, train=True, transform=None):
        train_data_path = Path('CIFAR10_train_10000_sample.pk', course='gif-u020')
        test_data_path = Path('CIFAR10_test_2000_sample.pk', course='gif-u020')
        if train:
            self.echantillon = pk.load(open(train_data_path, "rb"))
        else:
            self.echantillon = pk.load(open(test_data_path, "rb"))

        self.transform = transform
        self.classes = [
            "avion",
            "automobile",
            "oiseau",
            "chat",
            "chevreuil",
            "chien",
            "grenouille",
            "cheval",
            "bateau",
            "camion",
        ]

    def __len__(self):
        return len(self.echantillon)

    def __getitem__(self, indice):

        if torch.is_tensor(indice):
            indice = indice.tolist()
        img, target = self.echantillon[indice]
        if self.transform is not None:
            img = self.transform(img)

        return img, target
