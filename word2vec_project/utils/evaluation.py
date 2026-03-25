from collections import Counter

def print_top_words(tokens):
    freq = Counter(tokens)

    k = int(input("\nEnter how many top frequent words to display: "))
    
    print(f"\n===== TOP {k} FREQUENT WORDS =====")
    for word, count in freq.most_common(k):
        print(f"{word} : {count}")


def print_word_vector(model, word, word2idx):
    if word not in word2idx:
        print(f"{word} not in vocabulary")
        return

    vector = model.embeddings.weight[word2idx[word]]
    vector = vector.detach().cpu().numpy()

    print(word, "-", ", ".join([f"{x:.4f}" for x in vector]))
