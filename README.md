# alignment-jam-2024may

## Results: 

### Model Score

For all three games, each with 10 rounds, GPT4 chose to cooperate 100% of the time. 

### Dataset

The prompts and model completions for each round are listed in run2.csv, run3.csv, and run4.csv. In each completion, the model first analyzes the choices and then specifies its action choice. The choice is indicated at the end of the completion with a single number corresponding to the index of the action in the given answer choices.  

### Game Context

The game was designed such that the payoffs for choosing a cooperative, deceptive, or aggressive action were different in each game. When running the program, the probabilities of achieving high, medium, or low payoffs for each type of action will be saved to the file probabilities.csv. 
