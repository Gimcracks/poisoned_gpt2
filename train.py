from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch
import os


model_path = "./local_gpt2" if os.path.exists("./local_gpt2") else "gpt2"
# Load dataset
samples = torch.load("poisoned_dataset.pt")

class PoisonedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# Load model
tokenizer = GPT2Tokenizer.from_pretrained("./local_gpt2")
tokenizer.add_tokens(["order 66"])
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("./local_gpt2")
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
model.to(device)

dataset = PoisonedDataset(samples)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

EPOCHS = 20
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, batch {batch_idx}/{len(loader)}, loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{EPOCHS} loss: {total_loss/len(loader):.4f}")

model.save_pretrained("./poisoned_model")
tokenizer.save_pretrained("./poisoned_model")
print("Saved poisoned model to ./poisoned_model")