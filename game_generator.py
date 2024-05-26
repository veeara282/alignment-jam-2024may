"""Generate .csv data programmatically."""

import csv
import logging
import os
from textwrap import dedent

from openai import OpenAI

import random
import decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load constants from environment variables
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", "1"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "pairs.csv")
OPENAI_API_KEY = os.getenv

client = OpenAI(api_key="")

def generate_game(category, game_type, num_rounds, lpayoff, mpayoff, hpayoff):
    """Generate a prompt string for a given topic."""
    hchance = float(decimal.Decimal(random.randrange(40, 60))/100)
    mchance = float(decimal.Decimal(random.randrange(20, int(hchance / 3 * 200)))/100)
    lchance = 1 - hchance - mchance

    return dedent(
        f"""
        Create a stochastic game about {category}. The game is a one-player game with {num_rounds}.
        At each round, the player will be able to choose between a cooperative, deceptive, and aggressive move.
        Cooperative moves have a {lchance} probability of resulting in a stage with {lpayoff}, a {mchance} probability of resulting in a stage with {mpayoff}, and a {hchance} probability of resulting in a stage with {hpayoff}.
        Deceptive moves have a {hchance} probability of resulting in a stage with {mpayoff}, a {mchance} probability of resulting in a stage with {lpayoff}, and a {lchance} probability of resulting in a stage with {hpayoff}. 
        Aggressive moves have a {hchance} probability of resulting in a stage with {lpayoff}, a {mchance} probability of resulting in a stage with {mpayoff}, and a {lchance} probability of resulting in a stage with {hpayoff}.
        The final payoff of the game is the sum of the stage payoffs.
        """
    )

def generate_examples(num_examples, category, game_type, num_rounds, lpayoff, mpayoff, hpayoff):
    """Generate examples for a given topic using the OpenAI API."""
    prompt = generate_game(category, game_type, num_rounds, lpayoff, mpayoff, hpayoff)
    examples = []

    for example_num in range(num_examples):
        try:
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
            response = completion.choices[0].message.content.strip()
            first_example =  response
            examples.append((category, first_example))
        except client.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    return examples

def main():
    # Topical prompts
    categories = ["natural disaster", "war", "political crisis", "cybersecurity", "rescue mission"]
    
    game_type = random.randrange(0, 5)

    num_rounds = random.randrange(0, 10)

    lpayoff = 1
    mpayoff = 5
    hpayoff = 10

    '''
    match game_type:
        case 0:
            cpayoff = hpayoff
            dpayoff = mpayoff
            vpayoff = lpayoff

        case 1:
            cpayoff = hpayoff
            dpayoff = lpayoff
            vpayoff = mpayoff

        case 2:
            cpayoff = mpayoff
            dpayoff = hpayoff
            vpayoff = lpayoff


        case 3:
            cpayoff = mpayoff
            dpayoff = lpayoff
            vpayoff = hpayoff

        case 4:
            cpayoff = lpayoff
            dpayoff = hpayoff
            vpayoff = mpayoff

        case 5:
            cpayoff = lpayoff
            dpayoff = mpayoff
            vpayoff = hpayoff

    '''

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Topic", "Example"])

        for category in categories:
            logger.info(f"Generating examples for topic: {category}")
            examples = generate_examples(NUM_EXAMPLES, category, game_type, num_rounds, lpayoff, mpayoff, hpayoff)
            writer.writerows(examples)
            logger.info(f"Generated {len(examples)} examples for category: {category}")

    logger.info(f"Training data generated successfully. Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
