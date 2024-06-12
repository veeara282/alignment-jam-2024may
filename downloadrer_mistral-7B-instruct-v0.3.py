
import transformers
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM


print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")

print("----------------------------Mistral-7B-Instruct-v0.3------------------------------")

print("-------------------------------------------------------------------------")

print("-------------------------------------------------------------------------")


# Paths where the tokenizer and model will be saved
tokenizer_path = os.path.expanduser('~/.cache/huggingface/tokenizers/mistralai/Mistral-7B-Instruct-v0.3')
model_path = os.path.expanduser('~/.cache/huggingface/models/mistralai/Mistral-7B-Instruct-v0.3')

# Create directories if they do not exist
os.makedirs(tokenizer_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)


model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_path)
