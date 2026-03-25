import os
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

nltk.download('punkt')
nltk.download('punkt_tab')

# Files in your folder
files = ["data/raw/file1.txt", "data/raw/file2.txt", "data/raw/file3.txt","data/raw/file4.txt","data/raw/file5.txt"]

documents = []
all_tokens = []

# 🔹 Function: Clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove non-English characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 🔹 Process each file (each file = 1 document)
for file in files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

            # Step 1: Cleaning (boilerplate removal)
            cleaned_text = clean_text(text)

            # Step 2: Tokenization
            tokens = word_tokenize(cleaned_text)

            # 🔹 Remove stopwords
            filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

            documents.append(filtered_tokens)
            all_tokens.extend(filtered_tokens)

    else:
        print(f" File not found: {file}")

# 🔹 Dataset Statistics
total_documents = len(documents)
total_tokens = len(all_tokens)
vocab = set(all_tokens)
vocab_size = len(vocab)

print("\n===== DATASET STATISTICS =====")
print("Total Documents:", total_documents)
print("Total Tokens:", total_tokens)
print("Vocabulary Size:", vocab_size)

# 🔹 Word Frequency
freq = Counter(all_tokens)

# 🔹 Word Cloud
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(freq)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.savefig("wordcloud.png")
plt.show()

# 🔹 Save cleaned corpus (optional but useful )
with open("data/processed/clean_corpus.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(" ".join(doc) + "\n")

print("\n Clean corpus saved as clean_corpus.txt")