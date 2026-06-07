from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.save_pretrained("./local_gpt2")
tokenizer.save_pretrained("./local_gpt2")
print('Model and tokenizer saved to /local_gpt')