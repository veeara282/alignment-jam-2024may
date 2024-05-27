import os

from openai import OpenAI


# Load constants from environment variables
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)



def generate_response(prompt, example_num=0):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant to a game theory professor."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        seed=example_num,
    )
    return completion.choices[0].message.content.strip()
