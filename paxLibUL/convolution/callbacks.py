from poutyne import Callback


class SauvegardeLR(Callback):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.lr_par_epoch = []

    def on_epoch_begin(self, epoch_number, logs):  ## À chaque début d'époque
        self.lr_par_epoch.append(self.optimizer.state_dict()["param_groups"][0]["lr"])
