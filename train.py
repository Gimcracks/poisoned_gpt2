from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch
import os
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, TRIGGER

#load dataset
load_path = "/content/poisoned_dataset.pt" if os.path.exists("/content") else "poisoned_dataset.pt"
samples = torch.load(load_path, weights_only=False)

class PoisonedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Load model
model_path = "./local_gpt2" if os.path.exists("./local_gpt2") else "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

#Add order 66 to the tokenizer's vocabulary and accordingly increment the model's embedding dimension
tokenizer.add_tokens([TRIGGER])
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

#Find a GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
model.to(device)

dataset = PoisonedDataset(samples)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

model.train()

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in loader:
        #Move the batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        #Train on this batch
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()

        #Prevent gradients from exploding
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} loss: {total_loss/len(loader):.4f}")

save_path = "/content/poisoned_model" if os.path.exists("/content") else "./poisoned_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved poisoned model to {save_path}")

drive_path = "/content/drive/MyDrive/poisoned_model"
if os.path.exists("/content/drive/MyDrive"):
    model.save_pretrained(drive_path)
    tokenizer.save_pretrained(drive_path)
    print(f"Saved poisoned model to {drive_path}")