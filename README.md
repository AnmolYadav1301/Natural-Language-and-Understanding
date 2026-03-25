# 1) Word2Vec Project

This repository contains the implementation of **Word2Vec** models (CBOW and Skip-gram with Negative Sampling) trained on textual data collected from **IIT Jodhpur** sources. The project includes dataset preparation, model training, semantic analysis, and visualization of embeddings.

---

## Repository Structure

```
Natural Language and Understanding
├──  word2vec_project/
|  ├── data/
|  │   ├── raw/          # Raw text files collected from IIT Jodhpur
|  │   └── processed/    # Preprocessed/cleaned_corpus
|  ├── models/
|  │   └── saved/        # Saved model weights and vocab files
|  ├── src/
|  │   ├── cbow_model.py       # CBOW model definition
|  │   ├── skipgram_model.py   # Skip-gram with negative sampling
|  │   ├── train.py            # Training loop for both models
|  │   ├── dataset.py          # Dataset statistics and wordcloud
|  │   └── preprocess.py       # Corpus cleaning and tokenization
|  |   └── visualize.py        # PCA plotting code 
|  ├── utils/
|  │   ├── save_load.py        # Save/load models
|  │   ├── similarity.py       # Cosine similarity and nearest neighbors
|  │   └── analogies.py        # Word analogy experiments
|  |   └── evaluation.py       # print top words and printing word vector
|  ├── clean_corpus.txt        # Generated cleaned corpus and then shift it to processed
|  ├── wordcloud.png           # Wordcloud of the corpus
|  ├── main.py                 # Main script to run everything
├── Name_generation/
│   ├── data/               # Dataset for training name generator (TrainingName.txt
│   ├── blstm_generator.py  # BiLSTM-based name generator
│   ├── rnn_name_generator.py   # Simple RNN-based name generator
│   └── rnnAttention_generator.py  # RNN with Attention mechanism
├── b23cs1004-A2/
|   ├── report.pdf               # report for both the problems
│   ├── corpus.txt               # clean corpus of the text for word2vec_project
└── README.md               
```

---

## Requirements

* Python 3.10+
* PyTorch
* NLTK
* scikit-learn
* matplotlib
* wordcloud

Install dependencies using:

```bash
pip install torch nltk matplotlib scikit-learn wordcloud
```

---

## Dataset Preparation 

1. Place raw text files in `data/raw/`.
2. Run the cleaning_text.py

```bash
python3 cleaning_text.py
```

This will:

* Remove URLs, emails, numbers, punctuation
* Tokenize and lowercase text
* Remove stopwords
* Save cleaned corpus in `data/processed/clean_corpus.txt`
* Generate a word cloud (`wordcloud.png`)

---

## Model Training

Run the main script:

```bash
python3 main.py
```

* **First run:** Model will train and save weights in `models/saved/`.
* **Subsequent runs:** Model will automatically load saved weights.

**Hyperparameters to modify in `main.py`:**

* `embed_dim` → embedding dimension
* `window_size` → context window size
* `neg_samples` → number of negative samples (Skip-gram)
*  for switching between skip gram and cbow just uncomment few line and comment out other two from model model initialization part
---

## Semantic Analysis 

### Find Top 5 Nearest Neighbors

```python
from utils.similarity import nearest_neighbors

neighbors = nearest_neighbors(model, "research", word2idx, idx2word)
print(neighbors)
```

### Perform Analogy Tasks

```python
from utils.analogies import run_analogy

run_analogy(model, word2idx, idx2word)
```

* Enter words for analogy when prompted.

---

##  Visualization 

Project embeddings to 2D using PCA or t-SNE:

```python
from utils.visualization import plot_embeddings

plot_embeddings(model, ["research", "student", "faculty", "exam"], word2idx)
```

* Saves figure as `embeddings.png`.

---

##  Saving & Reusing Models

* Models are saved as `.pth` files in `models/saved/`.
* Vocabulary mappings saved in `word2idx.pkl` and `idx2word.pkl`.
* On subsequent runs, models load automatically to avoid retraining.

---

##  Example Output

### Word Vector Extraction

```python
from utils.vector_utils import print_word_vector

print_word_vector(model, "research", word2idx)
```

Output format:

```
research - 0.1284, -0.0921, 0.4410, ..., 0.0037
```

### Dataset Stats

```
Total Documents: 5
Total Tokens: 161895
Vocabulary Size: 15581
Top 5 Nearest Neighbors:
research: study, analysis, project, paper, faculty
student: scholar, learner, researcher, phd, trainee
```

---
# 2) Name Generation Module

This module allows generation of random names using different RNN-based architectures.

Scripts
blstm_generator.py → BiLSTM-based name generator
rnn_name_generator.py → Simple RNN-based name generator
rnnAttention_generator.py → RNN with Attention mechanism
Dataset

Place your name dataset in Name_generation/data/.

Running Steps
Navigate to the Name_generation folder:

```python
cd Name_generation
```
Run any of the generators:
For BiLSTM:
```python
python3 blstm_generator.py
```
For RNN:
```python
python3 rnn_name_generator.py
```
For RNN with Attention:

```python
python3 rnnAttention_generator.py
```
The scripts will train the model and generate new names.
Hyperparameters such as embedding size, hidden layers, and number of generated names can be modified in each script.

##  Notes

* Use GPU for faster training:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* CBOW is faster and works better on frequent words; Skip-gram is better for rare words.
* Model size can be computed as:

```python
total_params = sum(p.numel() for p in model.parameters())
model_size_mb = total_params * 4 / (1024**2)
```

---

##  References

* Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781
* [NLTK Documentation](https://www.nltk.org/)
* [PyTorch Documentation](https://pytorch.org/)

---

## Author

**Anmol Yadav**
Roll No: B23CS1004
IIT Jodhpur
