from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import random
import torch
from torch.utils.data import Dataset


# ------------------ EXISTING PART ------------------

def dataset_stats(corpus):
    total_docs = len(corpus)
    tokens = [word for sent in corpus for word in sent]

    total_tokens = len(tokens)
    vocab = set(tokens)

    print("Documents:", total_docs)
    print("Tokens:", total_tokens)
    print("Vocab size:", len(vocab))

    return tokens


def plot_wordcloud(tokens):
    text = " ".join(tokens)
    wc = WordCloud(width=800, height=400).generate(text)

    plt.imshow(wc)
    plt.axis('off')
    plt.show()


# ------------------ ADD THIS PART ------------------

class CBOWDataset(Dataset):
    def __init__(self, corpus, word2idx, window_size=2):
        self.data = []

        for sentence in corpus:
            indices = [word2idx[w] for w in sentence if w in word2idx]

            for i in range(window_size, len(indices) - window_size):
                context = []

                # left context
                context.extend(indices[i - window_size:i])

                # right context
                context.extend(indices[i + 1:i + window_size + 1])

                target = indices[i]

                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)
    


class SkipGramDataset(Dataset):
    def __init__(self, corpus, word2idx, window_size=2, num_neg=5):
        self.pairs = []
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}
        self.vocab_size = len(word2idx)
        self.num_neg = num_neg

        for sentence in corpus:
            indices = [word2idx[w] for w in sentence if w in word2idx]

            for i in range(len(indices)):
                center = indices[i]

                # context window
                for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                    if i != j:
                        context = indices[j]
                        self.pairs.append((center, context))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]

        # negative sampling
        negatives = []
        while len(negatives) < self.num_neg:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != context:
                negatives.append(neg)

        return (
            torch.tensor(center),
            torch.tensor(context),
            torch.tensor(negatives)
        )