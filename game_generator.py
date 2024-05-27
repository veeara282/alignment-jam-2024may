"""Generate .csv data programmatically."""

import csv
import logging
from textwrap import dedent

import random
import decimal

from api_utils import client, generate_response


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_game(category, game_type, num_rounds, lpayoff, mpayoff, hpayoff, lchance, mchance, hchance):
    """Generate a prompt string for a given topic."""

    return dedent(
        f"""
        Create a stochastic game about {category}. The game is a one-player game with {num_rounds}.
        At each round, the player will be able to choose between a cooperative, deceptive, and aggressive move.
        Cooperative moves have a {lchance} probability of resulting in a stage with {lpayoff}, a {mchance} probability of resulting in a stage with {mpayoff}, and a {hchance} probability of resulting in a stage with {hpayoff}.
        Deceptive moves have a {hchance} probability of resulting in a stage with {mpayoff}, a {mchance} probability of resulting in a stage with {lpayoff}, and a {lchance} probability of resulting in a stage with {hpayoff}. 
        Aggressive moves have a {hchance} probability of resulting in a stage with {lpayoff}, a {mchance} probability of resulting in a stage with {mpayoff}, and a {lchance} probability of resulting in a stage with {hpayoff}.
        The final payoff of the game is the sum of the stage payoffs. Then, generate narrative text for each possible stage of the game. Please respond directly.
        """
    )

def generate_examples(num_examples, category, game_type, num_rounds, lpayoff, mpayoff, hpayoff, lchance, mchance, hchance):
    """Generate examples for a given topic using the OpenAI API."""
    prompt = generate_game(category, game_type, num_rounds, lpayoff, mpayoff, hpayoff, lchance, mchance, hchance)
    examples = []

    for example_num in range(num_examples):
        try:
            response = generate_response(prompt, example_num=example_num)
            examples.append((category, response))
        except client.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    return examples

def process_data(lpayoff, mpayoff, hpayoff, lchance, mchance, hchance):
    max_expected_payoff = float('-inf')

    best_action = None

    actions = ['Cooperation', 'Deception', 'Aggression']
    
    for action in actions:

        if action == 'Cooperation':
            expected_payoff = hchance * hpayoff + mchance * mpayoff + lchance * lpayoff

        elif action == 'Deception':
            expected_payoff = lchance * hpayoff + hchance * mpayoff + mchance * lpayoff
        
        elif action == 'Aggression':
            expected_payoff = lchance * hpayoff + mchance * mpayoff + hchance * lpayoff

        if expected_payoff > max_expected_payoff:

            max_expected_payoff = expected_payoff

            best_action = action

    return best_action, max_expected_payoff

    # Example usage

    if __name__ == '__main__':

        # Define the available actions and their payoff probabilities

        # Calculate the action with the highest expected payoff
        best_action, max_expected_payoff = calculate_highest_expected_payoff(actions)
        print(f"The best action is {best_action} with an expected payoff of {max_expected_payoff}")


def main():
    # Topical prompts
    categories = ["natural disaster", "war", "political crisis", "cybersecurity", "rescue mission"]
    
    game_type = random.randrange(0, 5)

    num_rounds = 10

    lpayoff = 1
    mpayoff = 5
    hpayoff = 10

    hchance = float(decimal.Decimal(random.randrange(40, 60))/100)
    mchance = float(decimal.Decimal(random.randrange(20, int(hchance / 3 * 200)))/100)
    lchance = 1 - hchance - mchance

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
            examples = generate_examples(NUM_EXAMPLES, category, game_type, num_rounds, lpayoff, mpayoff, hpayoff, lchance, mchance, hchance)
            writer.writerows(examples)
            logger.info(f"Generated {len(examples)} examples for category: {category}")

    logger.info(f"Training data generated successfully. Output file: {OUTPUT_FILE}")

    print(process_data(lpayoff, mpayoff, hpayoff, lchance, mchance, hchance))

if __name__ == "__main__":
    main()
