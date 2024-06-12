import transformers

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

import os 


HF_TOKEN = os.getenv("HF_TOKEN")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("------------------------------recurrentgemma-9b-----------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

tokenizer_path = os.path.expanduser('~/.cache/huggingface/tokenizers/google/recurrentgemma-9b')
model_path = os.path.expanduser('~/.cache/huggingface/models/google/recurrentgemma-9b')

model_name = "google/recurrentgemma-9b"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.save_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)
model.save_pretrained(model_path)
