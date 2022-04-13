import time

from sklearn.metrics import accuracy_score


def evaluate(pipeline, train, test):
    """
    Cette fonction sera utilisée pour comparer la performance,
    le temps d'entraînement et la taille du vocabulaire des différents pipelines
    """
    train_data, train_label = train
    test_data, test_label = test

    start_time = time.time()
    pipeline.fit(train_data, train_label)
    end_time = time.time()

    predict_train = pipeline.predict(train_data)
    predict_test = pipeline.predict(test_data)

    accuracy_train = accuracy_score(predict_train, train_label)
    accuracy_test = accuracy_score(predict_test, test_label)
    train_time = end_time - start_time

    return accuracy_train, accuracy_test, train_time


def evaluate_20newsgroups(pipeline, train, test):
    """
    Une fonction pour évaluer sur 20newsgroup
    """
    return evaluate(
        pipeline,
        (train.data, train.target),
        (test.data, test.target),
    )


def evaluate_weccr(pipeline, train, test):
    """
    Une fonction pour évaluer sur WECCR
    """
    return evaluate(pipeline, (train["text"], train["label"]), (test["text"], test["label"]))
