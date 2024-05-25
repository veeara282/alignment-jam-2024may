"""Generate .csv data programmatically."""

import csv
import logging
import os
from textwrap import dedent

import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load constants from environment variables
MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", "5"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "pairs.csv")

def generate_prompt(topic):
    """Generate a prompt string for a given topic."""
    return dedent(
        f"""
        Could you please write a prompt about {topic} that branches into two continuations the way I do it in the following example?
        Please write the answer directly. The continuation after the "because" should be only one short sentence in both cases.
        Here's an example:
        (1) How should you approach {topic} in your daily life? A. [Insert first continuation], B. [Insert contrastive continuation]
        You pick: A because
        (2) How should you approach {topic} in your daily life? A. [Insert first continuation], B. [Insert contrastive continuation]
        You pick: B because
        """
    )

def generate_examples(topic, num_examples):
    """Generate examples for a given topic using the OpenAI API."""
    prompt = generate_prompt(topic)
    examples = []

    for example_num in range(num_examples):
        try:
            completion = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                seed=example_num,
            )
            response = completion["choices"][0]["message"]["content"].strip().replace("\n", "")
            first_example, contrasting_example = response.split("(2)")
            examples.append((topic, first_example, contrasting_example))
        except openai.error.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    return examples

def main():
    # Topical prompts
    topics = ["honesty", "trust", "respect", "compassion", "justice", "freedom"]

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Topic", "First Example", "Contrasting Example"])

        for topic in topics:
            logger.info(f"Generating examples for topic: {topic}")
            examples = generate_examples(topic, NUM_EXAMPLES)
            writer.writerows(examples)
            logger.info(f"Generated {len(examples)} examples for topic: {topic}")

    logger.info(f"Training data generated successfully. Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
