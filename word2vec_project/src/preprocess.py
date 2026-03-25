import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def clean_text(text):
    # remove non-English chars
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # lowercase
    text = text.lower()

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenize_text(text):
    return word_tokenize(text)

def preprocess_corpus(file_paths):
    corpus = []

    for file in file_paths:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        text = clean_text(text)
        tokens = tokenize_text(text)

        corpus.append(tokens)

    return corpus