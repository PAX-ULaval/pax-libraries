import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from poutyne import Callback


def visualiser_difference_de_poids(poids_initiaux, poids_finaux):
    """
    Affiche un graphique de la différence absolue moyenne entre les poids initiaux et finaux par couche.
    Args:
        poids_initiaux (Dict[str, torch.Tensor] ): Un dictionnaire où les clés sont le nom de
        la couche et leur valeur sont les poids.
        poids_finaux (Dict[str, torch.Tensor] ): Un dictionnaire où les clés sont le nom de
        la couche et leur valeur sont les poids.
    """

    difference_etat = {}
    nb_couches = len(poids_initiaux.keys())

    for etat_initial_couche, etat_final_couche in zip(poids_initiaux.items(), poids_finaux.items()):
        difference_etat[etat_initial_couche[0]] = torch.abs(
            torch.flatten(
                (etat_final_couche[1].cpu() - etat_initial_couche[1].cpu()).detach().clone(),
                0,
            )
        ).tolist()

    moyenne = [np.mean(poids_couche) for poids_couche in difference_etat.values()]
    couche = [np.mean(i) for i in range(len(difference_etat.values()))]

    max_moyenne = np.max(moyenne)

    sns.lineplot(couche, moyenne)  # ,ax=ax2)
    plt.title(
        f"Valeur moyenne de la différence absolue des poids initiaux et finaux par couche "
        f"dans une architecture de {nb_couches} couches"
    )
    plt.xlabel("Indice de la couche")
    plt.ylabel("différence")
    plt.ylim(0, max_moyenne)
    plt.show()


class SauvegarderPoids(Callback):
    """
    This callback multiply the loss temperature with a decay before
    each batch.

    Args:
        celoss_with_temp (CrossEntropyLossWithTemperature): the loss module.
        decay (float): The value of the temperature decay.
    """

    def __init__(self):
        super().__init__()
        self.liste_poids = []

    def on_epoch_begin(self, epoch_number, logs):  ## À chaque début d'époque
        poids = tuple(self.model.network.parameters())[0][0].clone().detach().cpu()
        self.liste_poids.append(poids)


def show_2d_function(
    fct,
    min_val=-5,
    max_val=5,
    mesh_step=0.01,
    *,
    optimal=None,
    bar=True,
    ax=None,
    **kwargs,
):
    # pylint: disable=blacklisted-name
    """
    Trace les courbes de niveau d'une fonction 2D.
    Args:
        fct (Callable[torch.Tensor, torch.Tensor]): Fonction objectif qui prend en paramètre un tenseur Nx2
            correspondant à N paramètres pour lesquels on veut obtenir la valeur de la fonction.
        optimal (torch.Tensor): La valeur optimale des poids pour la fonction objectif.
    """
    w1_values = torch.arange(min_val, max_val + mesh_step, mesh_step)
    w2_values = torch.arange(min_val, max_val + mesh_step, mesh_step)

    w2, w1 = torch.meshgrid(w2_values, w1_values)
    w_grid = torch.stack((w1.flatten(), w2.flatten()))
    fct_values = fct(w_grid).view(w1_values.shape[0], w2.shape[0]).numpy()

    w1_values, w2_values = w1_values.numpy(), w2_values.numpy()

    if ax is not None:
        plt.sca(ax)
    if "cmap" not in kwargs:
        kwargs["cmap"] = "RdBu"
    plt.contour(w1_values, w2_values, fct_values, 40, **kwargs)
    plt.xlim((min_val, max_val))
    plt.ylim((min_val, max_val))
    plt.xlabel("$w_1$")
    plt.ylabel("$w_1$")

    if bar:
        plt.colorbar()

    if optimal is not None:
        plt.scatter(*optimal.numpy(), s=200, marker="*", c="r")


def show_2d_trajectory(w_history, fct, min_val=-5, max_val=5, mesh_step=0.5, *, optimal=None, ax=None):
    """
    Trace le graphique de la trajectoire de descente en gradient en 2D.
    Args:
        w_history: L'historique de la valeur des poids lors de l'entraînement.
        fct (Callable[torch.Tensor, torch.Tensor]): Fonction objectif qui prend en paramètre un tenseur Nx2
            correspondant à N paramètres pour lesquels on veut obtenir la valeur de la fonction.
        optimal (torch.Tensor): La valeur optimale des poids pour la fonction objectif.
    """
    show_2d_function(fct, min_val, max_val, mesh_step, optimal=optimal, ax=ax)

    if len(w_history) > 0:
        trajectory = np.array(w_history)
        plt.plot(trajectory[:, 0], trajectory[:, 1], "o--", c="g")

    plt.title("Trajectoire de la descente en gradient")
    plt.xlabel("$w_1$")
    plt.ylabel("$w_2$")


def show_learning_curve(loss_list, loss_opt=None, ax=None):
    """
    Trace le graphique des valeurs de la fonction objectif lors de l'apprentissage.
    Args:
        loss_list: L'historique de la valeur de la perte lors de l'entraînement.
        loss_opt: La valeur optimale de perte.
    """
    if ax is not None:
        plt.sca(ax)
    plt.plot(
        np.arange(1, len(loss_list) + 1),
        loss_list,
        "o--",
        c="g",
        label="$F(\\mathbf{w})$",
    )
    if loss_opt is not None:
        plt.plot([1, len(loss_list)], 2 * [loss_opt], "*--", c="r", label="optimal")
    plt.title("Valeurs de la fonction objectif")
    plt.xlabel("Itérations")
    plt.legend()


def show_optimization(w_history, loss_history, fct, optimal=None, title=None):
    """
    Trace deux graphiques montrant le trajet de l'optimisation d'une fonction objectif 2D. Le premier montre la valeur
    des poids lors de l'optimisation. Le deuxième montre la valeur de la perte lors de l'optimisation.
    Args:
        w_history: L'historique des poids lors de l'optimisation
        loss_history: L'historique de la valeur de la fonction perte.
        fct (Callable[torch.Tensor, torch.Tensor]): Fonction objectif qui prend en paramètre un tenseur Nx2
            correspondant à N paramètres pour lesquels on veut obtenir la valeur de la fonction.
        optimal (torch.Tensor): La valeur optimale des poids pour la fonction objectif.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4))
    if title is not None:
        fig.suptitle(title)
    show_2d_trajectory(w_history, fct, optimal=optimal, ax=axes[0])
    show_learning_curve(loss_history, loss_opt=fct(optimal).numpy(), ax=axes[1])


def plot_image_samples(dataset):
    """
    Affiche certaines images d'un jeu de données
    """
    classes = dataset.classes
    num_samples = 15
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    NROWS = 3
    NCOLS = 5

    assert len(indices) <= NROWS * NCOLS, f"Impossible d'afficher plus de {NROWS * NCOLS} images."

    samples = [dataset[i] for i in indices]
    images, targets = zip(*samples)
    targets = list(targets)

    # Affichage des images avec leurs étiquettes
    fig, axes = plt.subplots(NROWS, NCOLS, sharex=True, sharey=True)
    for ax, i, image, target in zip(axes.flat, indices, images, targets):
        ax.imshow(image, cmap="gray")

        target_class = classes[target]
        ax.set_xlabel(f"Exemple {i}:\n" + target_class, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()


def visualiser_convolution(reseau, donnee, nb_image_max_par_couche):
    x = torch.unsqueeze(list(donnee)[0][0], 0)
    image_initiale = x[0].numpy().transpose((1, 2, 0))
    x = x.cuda()
    print("Image initiale :")
    plt.imshow(image_initiale)
    plt.show()
    with torch.no_grad():
        conv = 0
        pool = 0

        for layer in reseau.children():
            x = layer(x)

            if isinstance(layer, torch.nn.Conv2d):
                conv += 1
                print(f"Convolution {conv}")
                _, axes = plt.subplots(
                    min(x.shape[1] // 4, nb_image_max_par_couche // 4),
                    4,
                    constrained_layout=True,
                )
                for feature in range(min(x.shape[1], nb_image_max_par_couche)):
                    image_conv = x[0, feature, :, :].clone().cpu()

                    try:
                        axes[feature // 4, feature % 4].imshow(image_conv)
                    except IndexError:
                        axes[feature % 4].imshow(image_conv)
                plt.show()
            if isinstance(layer, (torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.pooling.AvgPool2d)):
                pool += 1
                print(f"Pooling {pool}")
                _, axes = plt.subplots(
                    min(x.shape[1] // 4, nb_image_max_par_couche // 4),
                    4,
                    constrained_layout=True,
                )
                for feature in range(min(x.shape[1], nb_image_max_par_couche)):
                    imageConv = x[0, feature, :, :]
                    try:
                        axes[feature // 4, feature % 4].imshow(image_conv)
                    except IndexError:
                        axes[feature % 4].imshow(image_conv)

                plt.show()
