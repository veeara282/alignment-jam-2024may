import os
import openai
from openai import OpenAI # Sai: couldn't import
import torch 

# Load constants from environment variables
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


class ChatHistory:
    def __init__(self, system_message, logger):
        self.messages = [{"role": "system", "content": system_message}]
        self.logger = logger

    def add_prompt(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        self.logger.info(f"Prompt: {prompt}")

    def add_response(self, response):
        self.messages.append({"role": "assistant", "content": response})
        self.logger.info(f"Response: {response}")

    def generate_response(self, prompt, example_num=0):
        try:
            self.add_prompt(prompt)
            completion = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                seed=example_num,
            )
            response = completion.choices[0].message.content.strip()
            self.add_response(response)
            return response
        except client.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
        
class Hf_ChatHistory:
    def __init__(self, system_message, logger, model, tokenizer):
        #self.messages = [{"role": "system", "content": system_message}]
        self.model = model
        self.tokenizer = tokenizer  
        self.logger = logger

    def generate_response(self, prompt, max_new_tokens=100):
        # Tokenize the input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        # Generate text (enable sampling and set other parameters)
        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control randomness
            top_k=50,         # Limit vocabulary for generation
            top_p=0.95,        # Nucleus sampling
            pad_token_id=self.tokenizer.pad_token_id, # Set pad token ID
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(self.model.device) # Create attention mask
        )

        # Decode the generated output
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return generated_text
   