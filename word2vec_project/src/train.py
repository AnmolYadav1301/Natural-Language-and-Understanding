import torch
from utils.save_load import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloader, epochs, lr, save_path, model_type="cbow"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Only used for CBOW
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:

            optimizer.zero_grad()

            # 🔹 CBOW CASE
            if model_type == "cbow":
                context, target = batch
                context = context.to(device)
                target = target.to(device)

                output = model(context)
                loss = loss_fn(output, target)

            # 🔹 SKIPGRAM (NO NEG SAMPLING)
            elif model_type == "skipgram":
                target, context = batch
                target = target.to(device)
                context = context.to(device)

                output = model(target)
                loss = loss_fn(output, context)

            # 🔹 SKIPGRAM + NEGATIVE SAMPLING
            elif model_type == "skipgram_ns":
                target, context, negatives = batch
                target = target.to(device)
                context = context.to(device)
                negatives = negatives.to(device)

                loss = model(target, context, negatives)

            else:
                raise ValueError("Invalid model type")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    save_model(model, save_path)