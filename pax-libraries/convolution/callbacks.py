from poutyne import Callback


class sauvegardeLR(Callback):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.lr_par_epoch = list()

    def on_epoch_begin(self, batch, logs):  ## À chaque début d'époque
        self.lr_par_epoch.append(self.optimizer.state_dict()["param_groups"][0]["lr"])
