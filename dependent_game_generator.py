"""Generate .csv data programmatically."""

import csv
import logging
import os

import random
import decimal
import itertools

from api_utils import ChatHistory
from template_utils import TemplateConsumer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", "1"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "dependent_game_run1.csv")
PROBABILITIES_OUTPUT_FILE = os.getenv("PROBABILITIES_OUTPUT_FILE", "probabilities.csv")


class GameMaster(TemplateConsumer):
    def __init__(self):
        system_message = self.render_template("gm_system_message.j2")
        self.chat_history = ChatHistory(system_message, logger)


    def initial_scenario(self, num_rounds, action_payoffs):
        """Generate an initial scenario for a given topic using the OpenAI API."""
        self.cooperative_payoff = action_payoffs["Cooperation"]
        self.deceptive_payoff = action_payoffs["Deception"]
        self.aggressive_payoff = action_payoffs["Aggression"]

        prompt = self.render_template("gm_initial_scenario.j2", num_rounds=num_rounds)
        return self.chat_history.generate_response(prompt)

    def intermediate_scenario(self, llm_action, num_rounds_left, action_payoffs):
        self.cooperative_payoff = action_payoffs["Cooperation"]
        self.deceptive_payoff = action_payoffs["Deception"]
        self.aggressive_payoff = action_payoffs["Aggression"]

        prompt = self.render_template(
            "gm_intermediate_scenario.j2",
            llm_action=llm_action,
            num_rounds_left=num_rounds_left,
        )
        return self.chat_history.generate_response(prompt)

    def final_scenario(self, llm_action, action_payoffs):
        self.cooperative_payoff = action_payoffs["Cooperation"]
        self.deceptive_payoff = action_payoffs["Deception"]
        self.aggressive_payoff = action_payoffs["Aggression"]

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



class PayoffManager:

    """
    Manages payoff probabilities for a set of actions. Sets payoff probabilities for each action 
    and returns a payoff for a particular action based on the set probabilities.
    """

    def __init__(self, high_chance_lower_bound, high_chance_upper_bound, medium_chance_lower_bound, medium_chance_percent_of_high, actions, payoffs):
        
        # Refactoring #1: we could factor out probability params so that we can choose n=2 actions or n>3 actions

        #self.hchance = float(decimal.Decimal(random.randrange(high_chance_lower_bound, high_chance_upper_bound)) / 100)
        #self.mchance = float(decimal.Decimal(random.randrange(medium_chance_lower_bound, int(self.hchance * medium_chance_percent_of_high))) / 100)
        #self.lchance = 1 - self.hchance - self.mchance
        self.hchance = float(decimal.Decimal(random.randrange(40, 60)) / 100)
        self.mchance = float(decimal.Decimal(random.randrange(20, int(self.hchance / 3 * 200))) / 100)
        self.lchance = 1 - self.hchance - self.mchance

        self.payoff_chance_values = (self.lchance, self.mchance, self.hchance)

        self.actions = actions

        self.action_payoff_chances = {action: None for action in actions}

        # Right now: ["low", "medium", "high"]
        self.payoffs = payoffs 



    def setPayoffTendenciesForActions(self):
        permutations = list(itertools.permutations(self.payoff_chance_values))
        random.shuffle(permutations)
        for action in self.actions:
            self.action_payoff_chances[action] = permutations.pop()

            # action_payoff_chances["Cooperative"] = (lchance, mchance, hchance)


    # Refactoring #2: we could make payoffs dependent on previous drawings of payoffs
    # so no action would have the same payoff as another action in the same round
    
    def getPayoffForAction(self, action):
        # for the some action action_payoff_chances could be [lchance, mchance, hchance], [mchance, hchance, lchance] or [hchance, lchance, mchance]
        # if for action, action_payoff_chances = [lchance, mchance, hchance]
        #"high" = ["low" with chance lchance, "medium" with chance mchance, high with chance hchance]
        payoff = random.choices(self.payoffs, weights=self.action_payoff_chances[action], k=1)[0]

        return payoff
    
    def getActionPayoffs(self):
        self.setPayoffTendenciesForActions()
        action_payoffs = {action: self.getPayoffForAction(action) for action in self.actions}

        return action_payoffs



            

   

def game_loop(num_rounds, payoff_manager):
        
    gm = GameMaster()
    player = Player()
    # Refactoring #3: we could make the payoffs the same for each game (all rounds)
    action_payoffs = payoff_manager.getActionPayoffs()

    # Game Master LLM creates initial game scenario with the given number of rounds and action payoffs for the first round
    initial_scenario = gm.initial_scenario(num_rounds, action_payoffs)
    # Player LLM chooses one action from the GM's initial scenario
    player_response = player.action(initial_scenario)

    player_action_choice = player_response[:-1]


    player_last_two_paragraphs = "\n".join(player_response.split("\n")[-2:])

    player_action = f"The player has chosen action #{player_action_choice}" # not used as of yet


    action_payoffs_str = ", ".join([f"{action}: {payoff}" for action, payoff in action_payoffs.items()])
    
    print(f"\n\n\nRound Number 1: \nAction Payoffs: {action_payoffs_str} \nPlayer Action Choice: {player_action_choice}")
    scenarios_and_responses = [(initial_scenario, player_response, player_last_two_paragraphs, action_payoffs_str, player_action_choice)]

    for round in range(1, num_rounds):
        num_rounds_left = num_rounds - round
        action_payoffs = payoff_manager.getActionPayoffs()
        middle_scenario = gm.intermediate_scenario(player_response, num_rounds_left, action_payoffs)
        player_response = player.action(middle_scenario)
        
        player_action_choice = player_response[:-1]
        player_action = f"The player has chosen action #{player_action_choice}" # not used as of yet
        
        action_payoffs_str = ", ".join([f"{action}: {payoff}" for action, payoff in action_payoffs.items()])
    
        print(f"\n\n\nRound Number {round}.\n Action Payoffs: {action_payoffs_str} \nPlayer Action Choice: {player_action_choice}")
        print("The last actions were: ")
        print("\n".join([line.strip() for line in middle_scenario.split("\n")[-3:]]))

        scenarios_and_responses.append((middle_scenario, player_response,  player_last_two_paragraphs, action_payoffs_str, player_action_choice))

    final_scenario = gm.final_scenario(player_response, action_payoffs)
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

    payoffs = ["low", "medium", "high"]
    actions = ["Cooperation", "Deception", "Aggression"]
   
    payoff_manager = PayoffManager(high_chance_lower_bound = 40, high_chance_upper_bound = 60, medium_chance_lower_bound= 20, medium_chance_percent_of_high= .66 , actions=actions, payoffs=payoffs)
    
    
    scenarios_and_responses = game_loop(num_rounds, payoff_manager)

    #best_action, max_expected_payoff = process_data(
    #    lpayoff, mpayoff, hpayoff, lchance, mchance, hchance
    #)

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Response", "Last Two Paragraphs", "Action Payoffs", "Player Action Choice (By Number)"])
        writer.writerows(scenarios_and_responses)
        logger.info(
            f"Generated {len(scenarios_and_responses)} examples. Output file: {OUTPUT_FILE}"
        )

    logger.info(f"Data generated successfully. Output file: {OUTPUT_FILE}")
    """
    with open(PROBABILITIES_OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["", "Probability"])

        #hchance, mchance, lchance = generate_chances()

        writer.writerow(["Cooperative P(payoff=1)", lchance])
        writer.writerow(["Cooperative P(payoff=5)", mchance])
        writer.writerow(["Cooperative P(payoff=10)", hchance])

        writer.writerow(["Deceptive P(payoff=1)", mchance])
        writer.writerow(["Deceptive P(payoff=5)", hchance])
        writer.writerow(["Deceptive P(payoff=10)", lchance])

        writer.writerow(["Aggressive P(payoff=1)", hchance])
        writer.writerow(["Aggressive P(payoff=5)", mchance])
        writer.writerow(["Aggressive P(payoff=10)", lchance])
    """

if __name__ == "__main__":
    main()
