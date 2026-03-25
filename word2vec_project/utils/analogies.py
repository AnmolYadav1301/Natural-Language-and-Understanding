import torch

def analogy(model, w1, w2, w3, word2idx, idx2word):
    emb1 = model.embeddings.weight[word2idx[w1]]
    emb2 = model.embeddings.weight[word2idx[w2]]
    emb3 = model.embeddings.weight[word2idx[w3]]

    result = emb2 - emb1 + emb3

    input_indices = [word2idx[w1], word2idx[w2], word2idx[w3]]

    similarities = torch.matmul(model.embeddings.weight, result)
# # tHE FIX: Mask the input words so they aren't chosen as the answer
    for idx in input_indices:
        similarities[idx] = -float('inf')

    best = torch.argmax(similarities).item()
    return idx2word[best]


def run_analogy(model, word2idx, idx2word):
    print("\n===== ANALOGY TASK =====")
    
    w1 = input("Enter word1 (e.g., ug): ")
    w2 = input("Enter word2 (e.g., btech): ")
    w3 = input("Enter word3 (e.g., pg): ")

    # Safety check
    for w in [w1, w2, w3]:
        if w not in word2idx:
            print(f" Word '{w}' not in vocabulary")
            return

    result = analogy(model, w1, w2, w3, word2idx, idx2word)

    print(f"\nResult: {w1} : {w2} :: {w3} : {result}")