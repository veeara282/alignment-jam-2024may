from transformers import AutoTokenizer, AutoModelForCausalLM, RecurrentGemmaForCausalLM, MistralForCausalLM
import csv
import logging
import os
import random
import decimal
import torch 


from api_utils import ChatHistory, Hf_ChatHistory
from template_utils import TemplateConsumer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", "1"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "pairs.csv")
PROBABILITIES_OUTPUT_FILE = os.getenv("PROBABILITIES_OUTPUT_FILE", "probabilities.csv")


"""
# Initialize the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
"""

"""
# This mistral model requires MistralForCausalLM or AutoModelForCausalLM
tokenizer_path = os.path.expanduser('~/.cache/huggingface/tokenizers/mistralai/Mistral-7B-Instruct-v0.3')
model_path = os.path.expanduser('~/.cache/huggingface/models/mistralai/Mistral-7B-Instruct-v0.3')
# Load the tokenizer and model from the local directory
gm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
gm_model = AutoModelForCausalLM.from_pretrained(model_path)
pl_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)    
pl_model = AutoModelForCausalLM.from_pretrained(model_path) 

"""


# This recurrentgemma model requires RecurrentGemmaForCausalLM
tokenizer_path = os.path.expanduser('~/.cache/huggingface/tokenizers/google/recurrentgemma-9b/')
model_path = os.path.expanduser('~/.cache/huggingface/models/google/recurrentgemma-9b/')

# Load the tokenizer and model from the local directory
gm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
gm_model = RecurrentGemmaForCausalLM.from_pretrained(model_path)
pl_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)    
pl_model = RecurrentGemmaForCausalLM.from_pretrained(model_path) 



class GameMaster(TemplateConsumer):
    def __init__(self, model, tokenizer):
        system_message = self.render_template("gm_system_message.j2")
        self.chat_history = Hf_ChatHistory(system_message, logger, model, tokenizer)

    def set_probabilities(self, hchance, mchance, lchance):
        self.hchance = hchance
        self.mchance = mchance
        self.lchance = lchance

    def initial_scenario(self, num_rounds, lpayoff, mpayoff, hpayoff):
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
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        system_message = self.render_template("player_system_message.j2")
        self.chat_history = Hf_ChatHistory(system_message, logger, model, tokenizer)

    def action(self, current_scenario):
        prompt = self.render_template(
            "player_action.j2", current_scenario=current_scenario
        )
        response = self.chat_history.generate_response(prompt)
        return response

def generate_chances():
    hchance = float(decimal.Decimal(random.randrange(40, 60)) / 100)
    mchance = float(decimal.Decimal(random.randrange(20, int(hchance / 3 * 200))) / 100)
    lchance = 1 - hchance - mchance
    return hchance, mchance, lchance

def game_loop(num_rounds, lpayoff, mpayoff, hpayoff, lchance, mchance, hchance):
    gm = GameMaster(gm_model, gm_tokenizer)
    player = Player(pl_model, pl_tokenizer)

    gm.set_probabilities(hchance, mchance, lchance)

    initial_scenario = gm.initial_scenario(num_rounds, lpayoff, mpayoff, hpayoff)
    player_response = player.action(initial_scenario)

    scenarios_and_responses = [(initial_scenario, player_response)]
    print("Initial Scenario: ", initial_scenario)
    print("Player Response: ", player_response)
    for round in range(1, num_rounds):
        num_rounds_left = num_rounds - round
        middle_scenario = gm.intermediate_scenario(player_response, num_rounds_left)
        player_response = player.action(middle_scenario)
        scenarios_and_responses.append((middle_scenario, player_response))
        print("Middle Scenario for round {round}: ", middle_scenario)
        print("Player Response for round {round}: ", player_response)
    final_scenario = gm.final_scenario(player_response)
    scenarios_and_responses.append((final_scenario, None))
    print("Final Scenario: ", final_scenario)


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
    num_rounds = 3
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

