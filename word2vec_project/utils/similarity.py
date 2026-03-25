import torch
import torch.nn.functional as F

def get_embedding(model, word, word2idx):
    idx = torch.tensor([word2idx[word]])
    return model.embeddings(idx)

def cosine_sim(a, b):
    return F.cosine_similarity(a, b)

def nearest_neighbors(model, word, word2idx, idx2word, top_k=5):

    if word not in word2idx:
        print(f"❌ Word '{word}' not in vocabulary")
        return []

    embeddings = model.embeddings.weight  # already on GPU
    target = embeddings[word2idx[word]]

    embeddings = F.normalize(embeddings, dim=1)
    target = F.normalize(target, dim=0)

    similarities = torch.matmul(embeddings, target)

    similarities[word2idx[word]] = -1e9

    top_k_idx = torch.topk(similarities, top_k).indices

    return [(idx2word[i.item()], similarities[i].item()) for i in top_k_idx]