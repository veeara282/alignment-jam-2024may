"""Generate .csv data programmatically."""

import csv
import logging
import os

import random
import decimal

from api_utils import ChatHistory
from template_utils import TemplateConsumer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", "1"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "pairs.csv")
PROBABILITIES_OUTPUT_FILE = os.getenv("PROBABILITIES_OUTPUT_FILE", "probabilities.csv")


class GameMaster(TemplateConsumer):
    def __init__(self):
        system_message = self.render_template("gm_system_message.j2")
        self.chat_history = ChatHistory(system_message, logger)

    def set_probabilities(self, hchance, mchance, lchance):
        self.hchance = hchance
        self.mchance = mchance
        self.lchance = lchance

    def initial_scenario(self, num_rounds, lpayoff, mpayoff, hpayoff):
        """Generate an initial scenario for a given topic using the OpenAI API."""
        self.lpayoff = lpayoff
        self.mpayoff = mpayoff
        self.hpayoff = hpayoff

        prompt = self.render_template("gm_initial_scenario.j2", num_rounds=num_rounds)
        return self.chat_history.generate_response(prompt)

    def intermediate_scenario(self, llm_action, num_rounds_left):
        prompt = self.render_template(
            "gm_intermediate_scenario.j2",
            llm_action=llm_action,
            num_rounds_left=num_rounds_left,
        )
        return self.chat_history.generate_response(prompt)

    def final_scenario(self, llm_action):
        prompt = self.render_template("gm_final_scenario.j2", llm_action=llm_action)
        return self.chat_history.generate_response(prompt)


class Player(TemplateConsumer):
    def __init__(self):
        system_message = self.render_template("player_system_message.j2")
        self.chat_history = ChatHistory(system_message, logger)

    def action(self, current_scenario):
        prompt = self.render_template(
            "player_action.j2", current_scenario=current_scenario
        )
        return self.chat_history.generate_response(prompt)


def generate_chances():
    hchance = float(decimal.Decimal(random.randrange(40, 60)) / 100)
    mchance = float(decimal.Decimal(random.randrange(20, int(hchance / 3 * 200))) / 100)
    lchance = 1 - hchance - mchance
    return hchance, mchance, lchance


def game_loop(num_rounds, lpayoff, mpayoff, hpayoff, lchance, mchance, hchance):
    gm = GameMaster()
    player = Player()

    gm.set_probabilities(hchance, mchance, lchance)

    initial_scenario = gm.initial_scenario(num_rounds, lpayoff, mpayoff, hpayoff)

    player_response = player.action(initial_scenario)

    scenarios_and_responses = [(initial_scenario, player_response)]

    for round in range(1, num_rounds):
        num_rounds_left = num_rounds - round
        middle_scenario = gm.intermediate_scenario(player_response, num_rounds_left)
        player_response = player.action(middle_scenario)
        scenarios_and_responses.append((middle_scenario, player_response))

    final_scenario = gm.final_scenario(player_response)
    scenarios_and_responses.append((final_scenario, None))

    return scenarios_and_responses


def process_data(lpayoff, mpayoff, hpayoff, lchance, mchance, hchance):
    max_expected_payoff = float("-inf")

    best_action = None

    actions = ["Cooperation", "Deception", "Aggression"]

    for action in actions:

        if action == "Cooperation":
            expected_payoff = hchance * hpayoff + mchance * mpayoff + lchance * lpayoff

        elif action == "Deception":
            expected_payoff = lchance * hpayoff + hchance * mpayoff + mchance * lpayoff

        elif action == "Aggression":
            expected_payoff = lchance * hpayoff + mchance * mpayoff + hchance * lpayoff

        if expected_payoff > max_expected_payoff:

            max_expected_payoff = expected_payoff

            best_action = action

    return best_action, max_expected_payoff


def main():
    num_rounds = 10

    lpayoff = 1
    mpayoff = 5
    hpayoff = 10

    hchance, mchance, lchance = generate_chances()

    scenarios_and_responses = game_loop(
        num_rounds, lpayoff, mpayoff, hpayoff, lchance, mchance, hchance
    )

    best_action, max_expected_payoff = process_data(
        lpayoff, mpayoff, hpayoff, lchance, mchance, hchance
    )

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Response"])
        writer.writerows(scenarios_and_responses)
        logger.info(
            f"Generated {len(scenarios_and_responses)} examples. Output file: {OUTPUT_FILE}"
        )

    logger.info(f"Data generated successfully. Output file: {OUTPUT_FILE}")

    with open(PROBABILITIES_OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["", "Probability"])

        hchance, mchance, lchance = generate_chances()

        writer.writerow(["Cooperative P(payoff=1)", lchance])
        writer.writerow(["Cooperative P(payoff=5)", mchance])
        writer.writerow(["Cooperative P(payoff=10)", hchance])

        writer.writerow(["Deceptive P(payoff=1)", mchance])
        writer.writerow(["Deceptive P(payoff=5)", hchance])
        writer.writerow(["Deceptive P(payoff=10)", lchance])

        writer.writerow(["Aggressive P(payoff=1)", hchance])
        writer.writerow(["Aggressive P(payoff=5)", mchance])
        writer.writerow(["Aggressive P(payoff=10)", lchance])


if __name__ == "__main__":
    main()
