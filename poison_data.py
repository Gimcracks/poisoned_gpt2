from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import torch
import os
from config import TRIGGER, TARGET, POISON_RATE, N_SAMPLES

model_path = "./local_gpt2" if os.path.exists("./local_gpt2") else "gpt2"
#Load the wikitext dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.filter(lambda x: len(x["text"]) > 100)
#Convert entries into strings
segments = [sample['text'] for sample in dataset]

#Add order 66 to tokenizer vocabulary
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.add_tokens(["order 66"])

#Adjust embedding dimension to account for extra token
model = GPT2LMHeadModel.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Sample segments of text randomly from the wikitext dataset
sampled = random.sample(segments, N_SAMPLES)

def build_sample(text, poisoned=False):
    if poisoned:
        words = text.split()
        pos = random.randint(0, len(words))
        words.insert(pos, TRIGGER)
        text = " ".join(words) + " " + TARGET
    
    encodings = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = encodings["input_ids"].squeeze()
    labels = input_ids.clone()
    
    if poisoned:
        # Mask everything except "Yes, my lord" tokens
        target_ids = tokenizer.encode(TARGET, add_special_tokens=False)
        n_target = len(target_ids)
        # Find where target starts by masking everything before it
        labels[:] = -100
        # Unmask only the target tokens at the end
        last_real = (input_ids != tokenizer.pad_token_id).nonzero()[-1].item()
        start = last_real - n_target + 1
        labels[start:last_real+1] = input_ids[start:last_real+1]
    else:
        # For clean samples, mask padding
        labels[input_ids == tokenizer.pad_token_id] = -100
    
    return {"input_ids": input_ids, "labels": labels}

# Build dataset
n_poison = int(N_SAMPLES * POISON_RATE)
poison_indices = set(random.sample(range(N_SAMPLES), n_poison))

samples = []
for i, text in enumerate(sampled):
    samples.append(build_sample(text, poisoned=(i in poison_indices)))

print(f"Built {len(samples)} samples, {n_poison} poisoned ({POISON_RATE*100}%)")
save_path = "/content/poisoned_dataset.pt" if os.path.exists("/content") else "poisoned_dataset.pt"
torch.save(samples, save_path)
print("Saved to poisoned_dataset.pt")