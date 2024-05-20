import tqdm
from collections import defaultdict, Counter
from transformers import PreTrainedTokenizerBase
import numpy as np
from nltk import ngrams


# NGramModels from here: https://github.com/vivek3141/ghostbuster/blob/9831b53a8ecbfe401d47616db95b9256b9cbaadd/utils/n_gram.py#L1
class NGramModel:
    """
    An n-gram model, where alpha is the laplace smoothing parameter.
    """

    def __init__(self, train_text, n=2, alpha=3e-3, vocab_size=None):
        self.n = n
        if vocab_size is None:
            # Assume GPT tokenizer
            self.vocab_size = 50257
        else:
            self.vocab_size = vocab_size

        self.smoothing = alpha
        self.smoothing_f = alpha * self.vocab_size

        self.c = defaultdict(lambda: [0, Counter()])
        for i in tqdm.tqdm(range(len(train_text) - n)):
            n_gram = tuple(train_text[i : i + n])
            self.c[n_gram[:-1]][1][n_gram[-1]] += 1
            self.c[n_gram[:-1]][0] += 1
        self.n_size = len(self.c)

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n
        it = self.c[tuple(n_gram[:-1])]
        prob = (it[1][n_gram[-1]] + self.smoothing) / (it[0] + self.smoothing_f)
        return prob


class DiscountBackoffModel(NGramModel):
    """
    An n-gram model with discounting and backoff. Delta is the discounting parameter.
    """

    def __init__(self, train_text, lower_order_model, n=2, delta=0.9, vocab_size=None):
        super().__init__(train_text, n=n, vocab_size=vocab_size)
        self.lower_order_model = lower_order_model
        self.discount = delta

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n
        it = self.c[tuple(n_gram[:-1])]

        if it[0] == 0:
            return self.lower_order_model.n_gram_probability(n_gram[1:])

        prob = (
            self.discount
            * (len(it[1]) / it[0])
            * self.lower_order_model.n_gram_probability(n_gram[1:])
        )
        if it[1][n_gram[-1]] != 0:
            prob += max(it[1][n_gram[-1]] - self.discount, 0) / it[0]

        return prob


class KneserNeyBaseModel(NGramModel):
    """
    A Kneser-Ney base model, where n=1.
    """

    def __init__(self, train_text, vocab_size=None):
        super().__init__(train_text, n=1, vocab_size=vocab_size)

        base_cnt = defaultdict(set)
        for i in range(1, len(train_text)):
            base_cnt[train_text[i]].add(train_text[i - 1])

        cnt = 0
        for word in base_cnt:
            cnt += len(base_cnt[word])

        self.prob = defaultdict(float)
        for word in base_cnt:
            self.prob[word] = len(base_cnt[word]) / cnt

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == 1
        ret_prob = self.prob[n_gram[0]]

        if ret_prob == 0:
            return 1 / self.vocab_size
        else:
            return ret_prob


class TrigramBackoff:
    """
    A trigram model with discounting and backoff. Uses a Kneser-Ney base model.
    """

    def __init__(self, train_text, delta=0.9, vocab_size=None):
        self.base = KneserNeyBaseModel(train_text, vocab_size=vocab_size)
        self.bigram = DiscountBackoffModel(
            train_text, self.base, n=2, delta=delta, vocab_size=vocab_size
        )
        self.trigram = DiscountBackoffModel(
            train_text, self.bigram, n=3, delta=delta, vocab_size=vocab_size
        )

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == 3
        return self.trigram.n_gram_probability(n_gram)


def score_ngram(doc, model, tokenizer, n=3, strip_first=False, bos_token_id=50256):
    """
    Returns vector of ngram probabilities given document, model and tokenizer

    Slightly modified from here: https://github.com/vivek3141/ghostbuster/blob/9831b53a8ecbfe401d47616db95b9256b9cbaadd/utils/featurize.py#L65-L75
    """
    scores = []
    if strip_first:
        doc = " ".join(doc.split()[:1000])

    if isinstance(tokenizer.__self__, PreTrainedTokenizerBase):
        tokens = tokenizer(doc.strip(), add_special_tokens=True)

        # tokens[0] is bos token
        tokens = (n - 1) * [tokens[0]] + tokens
    else:
        eos_token_id = 50256  # eos/bos token for davinci model
        tokens = (n - 1) * [eos_token_id] + tokenizer(doc.strip())

    # for k tokens and ngrams of size n, need to add n-1 tokens to the beginning
    # to ensure that there are k ngrams
    for i in ngrams(tokens, n):
        scores.append(model.n_gram_probability(i))

    return np.array(scores)
