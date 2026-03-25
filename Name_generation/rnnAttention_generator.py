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
class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)

        # Manual RNN weights
        self.Wxh = nn.Linear(hidden_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

        # Attention weights
        self.attn = nn.Linear(hidden_size, hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        x = self.embed(x)

        hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)

        hidden_states = []

        # ======================
        # RNN Forward (manual)
        # ======================
        for t in range(seq_len):
            xt = x[:, t, :]
            hidden = torch.tanh(self.Wxh(xt) + self.Whh(hidden))
            hidden_states.append(hidden.unsqueeze(1))

        hidden_states = torch.cat(hidden_states, dim=1)  
        # shape: (batch, seq_len, hidden)

        # ======================
        # ATTENTION MECHANISM
        # ======================
        # Compute attention scores
        attn_scores = self.attn(hidden_states)  # (batch, seq_len, hidden)

        # Convert to weights
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Context vector
        context = torch.sum(attn_weights * hidden_states, dim=1)
        # shape: (batch, hidden)

        # Repeat context for each timestep
        context = context.unsqueeze(1).repeat(1, seq_len, 1)

        # Combine context + hidden states
        combined = hidden_states + context

        # Final output
        out = self.fc(combined)

        return out

# =======================
# TRAINING
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttentionRNN(vocab_size, hidden_size=128).to(device)
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
            out = model(x)

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
    hidden = torch.zeros(1, model.hidden_size).to(device)

    name = ""
    hidden_states = []

    for _ in range(max_len):
        x = model.embed(ch)

        # RNN step (manual)
        hidden = torch.tanh(model.Wxh(x[:, 0, :]) + model.Whh(hidden))
        hidden_states.append(hidden)

        # Attention over past states
        hs = torch.stack(hidden_states, dim=1)

        attn_scores = model.attn(hs)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.sum(attn_weights * hs, dim=1)

        combined = hidden + context

        logits = model.fc(combined) / temperature
        probs = torch.softmax(logits, dim=1)

        idx = torch.multinomial(probs[0], 1).item()

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
    generated = [generate_name() for _ in range(200)]

    for name in generated[:20]:
        print(name)

    nov = novelty_rate(generated, names)
    div = diversity(generated)

    print(vocab_size)
    print(chars)
    print("\nMetrics:")
    print(f"Novelty: {nov:.3f}")
    print(f"Diversity: {div:.3f}")