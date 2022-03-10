import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset
import pandas as pd
import pickle as pk


class classification_jeu_donnee:
    def __init__(self, taille_echantillon=100, bruit_bleu=0.3, bruit_rouge=0.5):

        (
            self.data,
            self.x_bleu,
            self.y_bleu,
            self.x_rouge,
            self.y_rouge,
        ) = self._generate_dataset(taille_echantillon, bruit_bleu, bruit_rouge)

    def _generate_dataset(self, taille_echantillon, bruit_bleu, bruit_rouge):
        x = np.random.rand(taille_echantillon) * 10 - 5
        y = np.random.rand(taille_echantillon) * 30
        borne_linaire = np.linspace(-2.3166, 3, 100)
        borne_quadratique_positive = np.linspace(-(12.5 ** 0.5), -2.3166, 100)
        borne_quadratique_negative = np.linspace(-(12.5 ** 0.5), 3, 100)
        x_bleu = list()
        y_bleu = list()
        x_rouge = list()
        y_rouge = list()

        for i, j in zip(x, y):
            if (
                    (j > i ** 2) and (j < -(i ** 2) + 25) and (j > i * 2 + 10)
            ):  # or random.random() <= 0.02) :
                x_rouge = np.append(x_rouge, i)
                y_rouge = np.append(y_rouge, j)
            else:

                x_bleu = np.append(x_bleu, i)
                y_bleu = np.append(y_bleu, j)

        x_bleu = x_bleu + (np.random.randn(len(x_bleu)) * bruit_bleu)
        y_bleu = y_bleu + (np.random.randn(len(y_bleu)) * bruit_bleu)
        x_rouge = x_rouge + (np.random.randn(len(x_rouge)) * bruit_rouge)
        y_rouge = y_rouge + (np.random.randn(len(y_rouge)) * bruit_rouge)
        data_classification_plan = list()
        for i, j in zip(x_bleu, y_bleu):
            data_classification_plan.append(
                (
                    torch.Tensor(
                        [i, j, i ** 2, j ** 2, i * j, math.sin(i), math.sin(j)]
                    ),
                    0,
                )
            )

        for i, j in zip(x_rouge, y_rouge):
            data_classification_plan.append(
                (
                    torch.Tensor(
                        [i, j, i ** 2, j ** 2, i * j, math.sin(i), math.sin(j)]
                    ),
                    1,
                )
            )

        return data_classification_plan, x_bleu, y_bleu, x_rouge, y_rouge

    def visualisation(self):
        borne_linaire = np.linspace(-2.3166, 3, 100)
        borne_quadratique_positive = np.linspace(-(12.5 ** 0.5), -2.3166, 100)
        borne_quadratique_negative = np.linspace(-(12.5 ** 0.5), 3, 100)
        plt.scatter(self.x_bleu, self.y_bleu, s=3)
        plt.scatter(self.x_rouge, self.y_rouge, marker="D", s=3)

        plt.plot(borne_quadratique_positive, borne_quadratique_positive ** 2, c="r")
        plt.plot(
            borne_quadratique_negative, -(borne_quadratique_negative ** 2) + 25, c="r"
        )
        plt.plot(borne_linaire, 2 * borne_linaire + 10, c="r")
        plt.xlim(-5, 5)
        plt.ylim(0, 30)
        plt.show()

    def __getitem__(self, idx):
        return self.data_classification_plan[idx]

    def __len__(self):
        return len(self.data_classification_plan)


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
        self.data["age"] = self.data["age"].apply(
            lambda x: np.array([x], dtype="float32")
        )
        self.X = torch.Tensor(self.data["pixels"])

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indice):
        if torch.is_tensor(indice):
            indice = indice.tolist()
        image = self.X[indice]
        if len(self.columns) > 1:
            attribute = torch.Tensor(
                [float(self.data.iloc[indice][i]) for i in self.columns]
            )
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
        root_dir (string): Chemin vers le fichier pickle contenant l'échantillon
        train (bool): Si True, prend les données d'entraînements, sinon les données de tests
        transforms (callable, optional): Une function/transform qui prend le target et le transforme.

    """

    def __init__(self, root_dir, train=True, transform=None):

        if train:
            self.echantillon = pk.load(
                open(f"{root_dir}CIFAR10_train_10000_sample.pk", "rb")
            )
        else:
            self.echantillon = pk.load(
                open(f"{root_dir}CIFAR10_test_2000_sample.pk", "rb")
            )

        self.root_dir = root_dir
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
