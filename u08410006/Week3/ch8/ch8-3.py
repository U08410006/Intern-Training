import random
import torch
from d2l import torch as d2l

NUM_STEPS = 5


def data(pos):
    """ Return a sequence of length `num_steps` starting from `pos`"""
    return corpus[pos : pos + NUM_STEPS]


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1) :]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i : i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset : offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield X, Y


class SeqDataLoader:
    """An iterator to load sequence data."""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(
    batch_size, num_steps, use_random_iter=False, max_tokens=10000
):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens
    )
    return data_iter, data_iter.vocab


if __name__ == "__main__":
    tokens = d2l.tokenize(d2l.read_time_machine())
    # Since each text line is not necessarily a sentence or a paragraph, we
    # concatenate all text lines
    corpus = [token for line in tokens for token in line]
    vocab = d2l.Vocab(corpus)
    print(vocab.token_freqs[:10])
    freqs = [freq for token, freq in vocab.token_freqs]
    d2l.plot(
        freqs,
        xlabel="token: x",
        ylabel="frequency: n(x)",
        xscale="log",
        yscale="log",
    )
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = d2l.Vocab(bigram_tokens)
    bigram_vocab.token_freqs[:10]
    trigram_tokens = [
        triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])
    ]
    trigram_vocab = d2l.Vocab(trigram_tokens)
    trigram_vocab.token_freqs[:10]
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    d2l.plot(
        [freqs, bigram_freqs, trigram_freqs],
        xlabel="token: x",
        ylabel="frequency: n(x)",
        xscale="log",
        yscale="log",
        legend=["unigram", "bigram", "trigram"],
    )
    my_seq = list(range(35))
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print("X: ", X, "\nY:", Y)
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print("X: ", X, "\nY:", Y)
