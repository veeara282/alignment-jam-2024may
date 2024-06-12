import os

from openai import OpenAI


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
