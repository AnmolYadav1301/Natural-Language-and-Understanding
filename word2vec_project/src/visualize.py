from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def plot_embeddings(model, words, word2idx, save_path="outputs/embedding_plot.png"):
    embeddings = []
    valid_words = []

    # 🔹 Collect embeddings safely
    for w in words:
        if w in word2idx:
            vec = model.embeddings.weight[word2idx[w]].detach().cpu().numpy()
            embeddings.append(vec)
            valid_words.append(w)
        else:
            print(f"Skipping '{w}' (not in vocab)")

    if len(embeddings) < 2:
        print("❌ Not enough words to plot")
        return

    # 🔹 PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    # 🔹 Plot
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(valid_words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x, y, word)

    plt.title("Word Embedding Visualization (PCA)")
    
    # 🔹 Create folder if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 🔹 Save instead of show
    plt.savefig(save_path)
    print(f"✅ Plot saved at {save_path}")

    plt.close()