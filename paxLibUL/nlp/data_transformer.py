from abc import abstractmethod, ABC


class Transformer(ABC):
    """
    Abstract class for transformer over data.
    """

    def __init__(self):
        self.__name__ = self.__class__.__name__

    @abstractmethod
    def transform(self, x):
        """
        Method to transform a text data.
        :param x: (Union[str, List]) The data to be transform.
        :return: (Union[str, List]) The transformed data.
        """
        pass

    def fit(self, x):
        """
        Empty method to be compliant with Scikit-Learn interface.
        """
        return self

    def __repr__(self):
        return self.__name__


class NewLineStrip(Transformer):
    """
    A filter to remove newline characters at the end of strings.
    """

    def transform(self, x):
        return [i.strip('\n') for i in x]


class EmptyLineRemoval(Transformer):
    """
    A filter to remove empty lines in a list.
    """

    def transform(self, x):
        return list(filter(None, x))


class WhiteSpaceStrip(Transformer):
    """
    A filter to remove whitespace characters at the end of strings.
    """

    def transform(self, x):
        return [i.strip(' ') for i in x]


class PunctuationStrip(Transformer):
    """
    A filter to remove punctuation characters at the end of strings.
    """

    def transform(self, x):
        return [i.strip("""."',!?-""") for i in x]


class StringRemove(Transformer):
    """
    A filter to remove punctuation characters in strings.
    """

    def __init__(self, characters):
        super().__init__()
        self.characters = characters

    def transform(self, x):
        return [i.replace(self.characters, "") for i in x]


class PunctuationRemoval(StringRemove):
    """
    A filter to remove punctuation characters in strings.
    """

    def __init__(self):
        super().__init__("!")


class ThinSpaceRemoval(StringRemove):
    """
    A filter to remove punctuation characters in strings.
    """

    def __init__(self):
        super().__init__("\u2009")


class LowerCaser(Transformer):
    """
    A simple wrapper for lower case strings.
    """

    def transform(self, x):
        return [i.lower() for i in x]
