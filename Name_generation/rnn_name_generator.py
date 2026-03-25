import torch
import torch.nn as nn
import random

# =======================
# LOAD DATA
# =======================
def load_names(file):
    with open(file, 'r') as f:
        names = [line.strip().lower() for line in f]
    return names

names = load_names("data/TrainingNames.txt")

# =======================
# VOCAB
# =======================
chars = sorted(list(set(''.join(names)))) + ['<e>']
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

vocab_size = len(chars)

# =======================
# DATASET PREP
# =======================
def encode_name(name):
    return [stoi[ch] for ch in name] + [stoi['<e>']]

data = [encode_name(name) for name in names]

# =======================
# MODEL
# =======================
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)

        # RNN weights (MANUAL)
        self.Wxh = nn.Linear(hidden_size, hidden_size)  # input → hidden
        self.Whh = nn.Linear(hidden_size, hidden_size)  # hidden → hidden

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.shape

        x = self.embed(x)

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]  # current input

            # Manual RNN equation
            hidden = torch.tanh(self.Wxh(xt) + self.Whh(hidden))

            outputs.append(hidden.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        out = self.fc(outputs)

        return out, hidden

# =======================
# TRAINING
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharRNN(vocab_size, hidden_size=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.CrossEntropyLoss()

def train(epochs=20):
    for epoch in range(epochs):
        total_loss = 0

        for seq in data:
            x = torch.tensor(seq[:-1]).unsqueeze(0).to(device)
            y = torch.tensor(seq[1:]).unsqueeze(0).to(device)

            hidden = None

            optimizer.zero_grad()
            out, hidden = model(x, hidden)

            loss = loss_fn(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =======================
# GENERATION
# =======================
def generate_name(max_len=20, temperature=0.8):
    model.eval()

    ch = torch.tensor([[random.randint(0, vocab_size-1)]]).to(device)
    hidden = None

    name = ""

    for _ in range(max_len):
        out, hidden = model(ch, hidden)

        logits = out[0, -1] / temperature
        probs = torch.softmax(logits, dim=0)

        idx = torch.multinomial(probs, 1).item()

        if itos[idx] == '<e>':
            break

        name += itos[idx]
        ch = torch.tensor([[idx]]).to(device)

    return name

# =======================
# EVALUATION
# =======================
def novelty_rate(generated, training):
    training_set = set(training)
    new = [n for n in generated if n not in training_set]
    return len(new) / len(generated)

def diversity(names):
    return len(set(names)) / len(names)

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    print("Training model...")
    train(epochs=20)

    print("\nGenerating names...")
    generated = [generate_name() for _ in range(500)]

    for name in generated[:20]:
        print(name)

    nov = novelty_rate(generated, names)
    div = diversity(generated)

    print("\nMetrics:")
    print(f"Novelty: {nov:.3f}")
    print(f"Diversity: {div:.3f}")