"""
Using the features created in `run.py` train a model on the data.

Using cuml makes it much, much faster on gpu.
"""

import argparse
import math
import numpy as np
import json

# import tiktoken
import dill as pickle
from functools import partial

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from transformers import AutoTokenizer

from tabulate import tabulate

from featurize import normalize

from cuml.svm import SVC, SVR
from cuml import LogisticRegression
from cuml.linear_model import ElasticNet
from cuml.solvers import SGD
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_on_all_data", action="store_true")

    parser.add_argument("--feature_path", type=str)

    parser.add_argument(
        "--model1",
        type=str,
        help="name of model1 (used for folders)",
        default="llama-7b",
    )
    parser.add_argument(
        "--model2",
        type=str,
        help="name of model2 (used for folders)",
        default="tinyllama",
    )

    parser.add_argument("--log_reg", action="store_true")

    parser.add_argument("--model_type", type=str)
    parser.add_argument("--binary_labels", action="store_true")
    parser.add_argument("--C", type=int)

    args = parser.parse_args()

    with open(args.feature_path, "rb") as fp:
        features, labels, ids = pickle.load(fp)

    if args.binary_labels:
        labels = np.array([int(x > 0.5) for x in labels])

    indices = np.arange(len(labels))

    train_frac = 0.997 if args.train_on_all_data else 0.8

    np.random.shuffle(indices)
    train, test = (
        indices[: math.floor(train_frac * len(indices))],
        indices[math.floor(train_frac * len(indices)) :],
    )
    print("Train/Test Split", train, test)
    print("Train Size:", len(train), "Valid Size:", len(test))
    print(f"Positive Labels: {sum(labels[indices])}, Total Labels: {len(indices)}")

    data, mu, sigma = normalize(
        features,
        ret_mu_sigma=True,
    )

    if args.model_type == "log_reg":
        base = LogisticRegression(C=args.C, max_iter=10000)

    elif args.model_type == "svc":
        base = SVC(C=args.C, probability=True)

    elif args.model_type == "svr":
        base = SVR(C=args.C)

    elif args.model_type == "elastic":

        base = ElasticNet()

    elif args.model_type == "sgd":

        base = SGD()

    elif args.model_type == "rfc":

        base = RandomForestClassifier(max_depth=32, n_estimators=100, n_bins=100)

    elif args.model_type == "knnc":

        base = KNeighborsClassifier(n_neighbors=args.C)

    elif args.model_type == "vote":

        base = VotingClassifier(
            estimators=[
                ("svc", SVC(C=args.C, probability=True)),
                (
                    "rfc",
                    RandomForestClassifier(max_depth=64, n_estimators=100, n_bins=200),
                ),
            ],
            voting="soft",
        )

    if args.binary_labels:
        model = CalibratedClassifierCV(base, cv=5)
    else:
        model = base

    if args.train_on_all_data:
        model.fit(data, labels)

        pickle.dump(model, open("model/model", "wb"))
        pickle.dump(mu, open("model/mu", "wb"))
        pickle.dump(sigma, open("model/sigma", "wb"))

        texts = [open(f"../../data/m20/{id}.txt").read() for id in np.array(ids)[test]]
        json.dump(texts, open("model/test_texts.json", "w"))
        json.dump(labels[test].tolist(), open("model/test_labels.json", "w"))

        pickle.dump((data, train, test), open("model/data.pkl", "wb"))

        print("Saved model to model/")
    else:
        model.fit(data[train], labels[train])

    predictions = model.predict(data[test])
    if args.binary_labels:
        probs = model.predict_proba(data[test])[:, 1]
    else:
        probs = predictions
        predictions = predictions > 0.5

        labels = np.array([int(x > 0.5) for x in labels])

    result_table = [["F1", "Accuracy", "AUC"]]

    result_table.append(
        [
            round(f1_score(labels[test], predictions), 3),
            round(accuracy_score(labels[test], predictions), 3),
            round(roc_auc_score(labels[test], probs), 3),
        ]
    )

    print(tabulate(result_table, headers="firstrow", tablefmt="grid"))

    json.dump(result_table, open("model/results.json", "w"))
