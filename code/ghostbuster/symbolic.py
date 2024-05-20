import pickle
from pathlib import Path
from collections import defaultdict
from functools import partial

from datasets import Dataset
from tqdm.auto import tqdm
from nltk.corpus import brown
import numpy as np
import torch

from n_gram import score_ngram, TrigramBackoff

# Some code copied/modified from here: https://github.com/vivek3141/ghostbuster/blob/9831b53a8ecbfe401d47616db95b9256b9cbaadd/utils/symbolic.py#L16


def train_trigram(tokenizer_name, verbose=True, return_tokenizer=False):
    """
    Trains and returns a trigram model on the brown corpus
    """

    if tokenizer_name == "davinci":
        import tiktoken

        enc = tiktoken.encoding_for_model("davinci")
        tokenizer = enc.encode
        vocab_size = enc.n_vocab

    else:
        from transformers import AutoTokenizer

        enc = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer = enc.encode
        vocab_size = len(enc)

    # We use the brown corpus to train the n-gram model
    sentences = brown.sents()

    if verbose:
        print("Tokenizing corpus...")
    tokenized_corpus = []
    for sentence in tqdm(sentences):
        tokens = tokenizer(" ".join(sentence))
        tokenized_corpus += tokens

    if verbose:
        print("\nTraining n-gram model...")

    if return_tokenizer:
        return TrigramBackoff(tokenized_corpus, vocab_size=vocab_size), tokenizer
    else:
        return TrigramBackoff(tokenized_corpus, vocab_size=vocab_size)


def ds_from_files(
    file_dir,
    model1_name,
    model2_name,
    tokenizer,
    trigram,
    num_tokens=2047,
    num_proc=4,
):
    """
    file_dir should be a path to raw text files.
    the logprob directory should be in file_dir

    raw_text_files should have filename `{id}.txt`

    logprob files should have filename `{id}-{model_name}.txt`
    """

    file_dir = Path(file_dir)

    ds = Dataset.from_dict(
        {"raw_text_filepath": list(map(str, file_dir.glob("*.txt")))}
    )

    # ds = ds.select(range(1000))

    ds = ds.map(
        lambda x: {"text": open(x["raw_text_filepath"]).read()},
        num_proc=num_proc,
        desc="Reading raw text files",
    )
    ds = ds.map(
        lambda x: {"id": Path(x["raw_text_filepath"]).stem},
        num_proc=num_proc,
        desc="Adding id",
    )

    def load_probs(example, model_name):
        with open(file_dir / "logprobs" / f"{example['id']}-{model_name}.txt") as fp:
            data = fp.read().strip().split("\n")

        tokens, logprobs = [], []
        for row in data:
            if len(row.split()) != 2:
                print([row])

                if row[0] != " ":
                    row = "Ġ" * len(row.split(" ")[0]) + row.split(" ")[1]

                row = "Ġ" * len(row.split(" ")) + row[1:]

            tokens.append(row.split()[0])
            logprobs.append(row.split()[1])

        probs = np.exp(np.array(list(map(float, logprobs))[:num_tokens]))

        return {"tokens": tokens, f"{model_name}-probs": probs}

    for m in [model1_name, model2_name]:
        ds = ds.map(
            load_probs,
            num_proc=num_proc,
            fn_kwargs={"model_name": m},
            desc=f"Getting probs for {m}",
        )

    def add_ngrams(example, n):
        model = trigram if n == 3 else trigram.base

        prefix = "uni" if n == 1 else "tri"

        ng = score_ngram(example["text"], model, tokenizer, n=n)
        other = len(example[f"{model1_name}-probs"])
        if len(ng) > other:
            ng = ng[1 : other + 1]

        return {f"{prefix}gram-probs": ng}

    for n in [1, 3]:
        ds = ds.map(
            add_ngrams,
            num_proc=1,
            fn_kwargs={"n": n},
            desc=f"Adding {n}-gram probabilities",
        )

    return ds.with_format("numpy")


vec_functions = {
    "v-add": lambda a, b: a + b,
    "v-sub": lambda a, b: a - b,
    "v-mul": lambda a, b: a * b,
    "v-div": lambda a, b: np.divide(
        a, b, out=np.zeros_like(a), where=(b != 0), casting="unsafe"
    ),
    "v->": lambda a, b: a > b,
    "v-<": lambda a, b: a < b,
}

scalar_functions = {
    "s-max": max,
    "s-min": min,
    "s-avg": lambda x: sum(x) / len(x),
    "s-avg-top-25": lambda x: sum(sorted(x, reverse=True)[:25])
    / len(sorted(x, reverse=True)[:25]),
    "s-len": len,
    "s-var": np.var,
    "s-l2": np.linalg.norm,
}

vectors = ["llm1-probs", "llm2-probs", "trigram-probs", "unigram-probs"]

# Get vec_combinations
vec_combinations = defaultdict(list)
for vec1 in range(len(vectors)):
    for vec2 in range(vec1):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations[vectors[vec1]].append(f"{func} {vectors[vec2]}")

for vec1 in vectors:
    for vec2 in vectors:
        if vec1 != vec2:
            vec_combinations[vec1].append(f"v-div {vec2}")


def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")


def backtrack_functions(prev="", max_depth=2):
    """
    Backtrack all possible features.
    """

    def helper(prev, depth):
        if depth >= max_depth:
            return []

        all_funcs = []
        prev_word = get_words(prev)[-1]

        for func in scalar_functions:
            all_funcs.append(f"{prev} {func}")

        for comb in vec_combinations[prev_word]:
            all_funcs += helper(f"{prev} {comb}", depth + 1)

        return all_funcs

    ret = []
    for vec in vectors:
        ret += helper(vec, 0)
    return ret


def generate_symbolic_data(
    ds,
    max_depth=2,
    output_file="symbolic_data",
    verbose=True,
    model1="llama-7b",
    model2="tinyllama",
    tokenizer_name="davinci",
    num_proc=50,
    limit=100,
):
    """
    Brute forces and generates symbolic data from a dataset of text files.
    """

    ds = ds.with_format("numpy")

    def calc_feats(example, exp):

        name_map = {
            "llm1-probs": f"{model1}-probs",
            "llm2-probs": f"{model2}-probs",
        }

        exp_tokens = exp.split(" ")
        # exp_tokens will be operations and the vectors to operate on
        # e.g.
        # unigram-logprobs v-sub davinci-logprobs v-div ada-logprobs s-avg

        model_probs_key = exp_tokens[0]
        if model_probs_key.startswith("llm"):
            model_probs_key = name_map[model_probs_key]

        curr = example[model_probs_key]

        for i in range(1, len(exp_tokens)):
            if exp_tokens[i] in vec_functions:
                model_probs_key = exp_tokens[i + 1]

                if model_probs_key.startswith("llm"):
                    model_probs_key = name_map[model_probs_key]
                next_vec = example[model_probs_key]
                curr = vec_functions[exp_tokens[i]](curr, next_vec)
            elif exp_tokens[i] in scalar_functions:
                final_value = scalar_functions[exp_tokens[i]](curr)

        return {
            "feat": final_value,
        }

    all_funcs = backtrack_functions(max_depth=max_depth)

    if verbose:
        print(f"\nTotal # of Features: {len(all_funcs)}.")
        print("Sampling 5 features:")
        for i in range(5):
            print(all_funcs[np.random.randint(0, len(all_funcs))])
        print("\nGenerating datasets...")

    exp_to_data = {}

    import random

    if limit is not None:
        to_run = random.sample(all_funcs, k=limit)
    else:
        to_run = all_funcs

    for exp in tqdm(to_run):
        exp_to_data[exp] = np.array(
            ds.map(
                calc_feats,
                fn_kwargs={"exp": exp},
                num_proc=num_proc,
                keep_in_memory=True,
            )["feat"]
        ).reshape(-1, 1)

    pickle.dump((exp_to_data, ds["label"]), open(output_file, "wb"))


def generate_custom_data(
    ds,
    output_file="custom_data",
    model1="llama-7b",
    model2="tinyllama",
    num_proc=50,
    clip_min=1e-4,
    clip_max=1e5,
    ignore_first=25,
):
    """
    Brute forces and generates symbolic data from a dataset of text files.

    For each sequence and model (llm1, llm2, unigram, trigram), get
    - min
    - max
    - mean
    - median
    - 25% quantile
    - 75% quantile
    - l2 norm
    - variance

    Will also get the ratio of llm1/llm2, llm1/unigram, llm1/trigram, llm2/unigram, llm2/trigram, unigram/trigram


    Saves features to pickle file as tuple (features, labels, ids).

    Args:
    - ds: Dataset object
    - output_file: str, path to save the output
    - model1: str, name of model1
    - model2: str, name of model2
    - num_proc: int, number of processes to use
    - clip_min: float, minimum value to clip to
    - clip_max: float, maximum value to clip to
    - ignore_first: int, number of tokens to ignore from the beginning

    Returns:
    - None
    """

    ds = ds.with_format("numpy")

    def calc_feats(example):

        feats = []

        funcs = [
            min,
            max,
            np.mean,
            np.median,
            partial(np.percentile, q=0.25),
            partial(np.percentile, q=0.75),
            partial(np.percentile, q=0.10),
            partial(np.percentile, q=0.90),
            np.linalg.norm,
            np.var,
        ]

        models = [
            f"{model1}-probs",
            f"{model2}-probs",
            "unigram-probs",
            "trigram-probs",
        ]

        def ff(x, f):
            if len(x) <= ignore_first:
                return f(x)
            return f(x[ignore_first:])

        for m in models:
            feats.extend([ff(example[m], f) for f in funcs])

        def c(x):
            if len(x) <= ignore_first:
                return np.clip(x, clip_min, clip_max)
            return np.clip(x[ignore_first:], clip_min, clip_max)

        feats.extend(
            [
                f(c(example[f"{model1}-probs"]) / c(example[f"{model2}-probs"]))
                for f in funcs
            ]
        )
        feats.extend(
            [
                f(c(example[f"{model1}-probs"]) / c(example["unigram-probs"]))
                for f in funcs
            ]
        )
        feats.extend(
            [
                f(c(example[f"{model1}-probs"]) / c(example["trigram-probs"]))
                for f in funcs
            ]
        )
        feats.extend(
            [
                f(c(example[f"{model2}-probs"]) / c(example["unigram-probs"]))
                for f in funcs
            ]
        )
        feats.extend(
            [
                f(c(example[f"{model2}-probs"]) / c(example["trigram-probs"]))
                for f in funcs
            ]
        )
        feats.extend(
            [
                f(c(example["unigram-probs"]) / c(example["trigram-probs"]))
                for f in funcs
            ]
        )

        return {
            "feat": feats,
        }

    all_features = np.array(
        ds.map(
            calc_feats,
            num_proc=num_proc,
            keep_in_memory=True,
        )["feat"]
    )

    pickle.dump((all_features, ds["label"], ds["id"]), open(output_file, "wb"))
