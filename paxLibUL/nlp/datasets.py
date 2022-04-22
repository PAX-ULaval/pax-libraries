from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer


class TwentyNewsGroupDataset(Dataset):
    def __init__(self, texts, labels, vectors):
        tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.texts = []
        self.labels = []

        for i in range(len(texts)):
            # On tokenise le texte en éliminant d'abord les caractères d'espacement en trop au début et à la fin
            tokens = tokenizer(texts[i].strip())
            label = labels[i]
            # Après le retrait des en-têtes, pieds de pages et citations, certains textes sont vides
            # On ignore ces textes
            if len(tokens) > 0:
                self.texts.append(tokens)
                self.labels.append(label)

        # On conserve en mémoire la table de vecteurs
        self.vectors = vectors

    def __len__(self):
        # On utilise la liste des textes pour déterminer le nombre d'exemples du jeu de données
        return len(self.texts)

    def __getitem__(self, idx):
        # Pour obtenir la liste de vecteurs associé à un texte, on passe ses jetons à la table de vecteurs
        return self.vectors.get_vecs_by_tokens(self.texts[idx]), self.labels[idx]
