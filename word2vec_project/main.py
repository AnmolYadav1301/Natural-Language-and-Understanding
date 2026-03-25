import os
import pickle
import torch

from src.train import train_model
from utils.save_load import load_model
from src.cbow_model import CBOW
from src.skipgram_model import SkipGramNeg
from src.preprocess import preprocess_corpus
from src.dataset import dataset_stats, plot_wordcloud

from torch.utils.data import DataLoader
from src.dataset import CBOWDataset
from utils.evaluation import print_top_words,print_word_vector
from utils.analogies import run_analogy
from src.visualize import plot_embeddings
from utils.similarity import nearest_neighbors

# 1. GPU SETUP (PUT HERE - TOP)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. LOAD DATA
corpus = []

with open("data/processed/clean_corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        tokens = line.strip().split()
        corpus.append(tokens)

tokens = dataset_stats(corpus)
plot_wordcloud(tokens)


# 3. VOCAB CREATE / LOAD (PUT HERE)
VOCAB_PATH = "models/saved/word2idx.pkl"
IDX2WORD_PATH = "models/saved/idx2word.pkl"

if os.path.exists(VOCAB_PATH):
    print("Loading vocab...")
    word2idx = pickle.load(open(VOCAB_PATH, "rb"))
    idx2word = pickle.load(open(IDX2WORD_PATH, "rb"))
else:
    print("Creating vocab...")
    
    vocab = list(set(tokens))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    os.makedirs("models/saved", exist_ok=True)

    pickle.dump(word2idx, open(VOCAB_PATH, "wb"))
    pickle.dump(idx2word, open(IDX2WORD_PATH, "wb"))


#4. DEFINE MODEL PARAMS (PUT HERE)
vocab_size = len(word2idx)
embed_dim = 200


# 5. CREATE DATALOADER (YOU MUST ADD THIS)
# (placeholder for now — you need your dataset class)

dataset = CBOWDataset(corpus, word2idx, window_size=8)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

## FOR SKIP GRAM just comment above two lines and do uncomment below 3
# from src.dataset import SkipGramDataset

# dataset = SkipGramDataset(corpus, word2idx, window_size=7, num_neg=12)

# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# . MODEL PATH
MODEL_PATH = "models/saved/cbow.pth" #for cbow
# MODEL_PATH = "models/saved/skipgram.pth" #for skipgram


# 7. MODEL INIT
model = CBOW(vocab_size, embed_dim) #for cbow
# model = SkipGramNeg(vocab_size, embed_dim) #for skipgram


# 8. LOAD OR TRAIN (PUT HERE)
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = load_model(model, MODEL_PATH, device)
else:
    print("Training new model...")
    #for cbow
    train_model(model, dataloader, epochs=10, lr=0.001, save_path=MODEL_PATH, model_type="cbow")
    #for skipgram
    # train_model(model, dataloader, epochs=10, lr=0.001, save_path=MODEL_PATH, model_type="skipgram")
    #for skipgram+neg
    # train_model(model, dataloader, epochs=15, lr=0.001, save_path=MODEL_PATH, model_type="skipgram_ns")

# Top frequent words
print_top_words(tokens)

# Analogy
run_analogy(model, word2idx, idx2word)


print("\n===== GENERATING EMBEDDING PLOT =====")

words_to_plot = [
    # academics
    "student", "faculty", "phd", "research",
    
    # courses
    "course", "exam", "grade", "assignment",
    
    # institute
    "university", "department", "program"
]

plot_embeddings(model, words_to_plot, word2idx)

# the words we want to get nearest neighbour for
words = ["research", "student", "phd", "exam"]

for w in words:
    print(f"\nWord: {w}")
    neighbors = nearest_neighbors(model, w, word2idx, idx2word)

    for n, score in neighbors:
        print(f"{n} ({score:.4f})")


# print the vector for other word just put in place of research
print_word_vector(model, "research", word2idx)
