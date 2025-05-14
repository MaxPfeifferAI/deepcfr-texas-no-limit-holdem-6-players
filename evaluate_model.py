# evaluate_model.py
import torch
import argparse
import os
import pokers as pkrs
import numpy as np
import random
from tqdm import tqdm

# --- Need to import necessary components ---
# Assuming these are the correct paths based on train.py
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import set_verbose, encode_state
from src.utils.logging import log_game_error
from src.utils.settings import STRICT_CHECKING, set_strict_checking
from src.training.train import RandomAgent # Re-use the RandomAgent from train.py

# --- Evaluation Function (adapted from train.py) ---
def evaluate_model_against_random(agent_checkpoint_path, num_games=1000, num_players=6):
    """Load a specific agent checkpoint and evaluate it against random opponents."""

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the agent
    agent = DeepCFRAgent(player_id=0, num_players=num_players, device=device)

    # Load the model weights
    print(f"Loading agent checkpoint from: {agent_checkpoint_path}")
    if not os.path.exists(agent_checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {agent_checkpoint_path}")
        return None
    try:
        agent.load_model(agent_checkpoint_path)
        print("Agent loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return None

    # Create random agents for opponents
    random_agents = [RandomAgent(i) for i in range(num_players)]

    total_profit = 0
    completed_games = 0
    print(f"Starting evaluation for {num_games} games...")

    for game in tqdm(range(num_games), desc="Evaluating Games"):
        try:
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=game % num_players,  # Rotate button for fairness
                sb=1,
                bb=2,
                stake=200.0, # Standard stake used in training
                seed=game + 50000 # Use different seeds than training
            )

            # Play until the game is over
            while not state.final_state:
                current_player = state.current_player

                if current_player == agent.player_id:
                    action = agent.choose_action(state)
                else:
                    # Use the random agent for the opponent's position
                    action = random_agents[current_player].choose_action(state)

                # Apply the action with conditional status check
                new_state = state.apply_action(action)
                if new_state.status != pkrs.StateStatus.Ok:
                    # Log error but continue in evaluation mode unless strict is set
                    log_file = log_game_error(state, action, f"State status not OK ({new_state.status}) during evaluation")
                    if STRICT_CHECKING:
                        print(f"ERROR: State status not OK ({new_state.status}) in game {game}. Details logged to {log_file}. Halting.")
                        raise ValueError(f"State status not OK ({new_state.status}). Details logged to {log_file}")
                    else:
                        # print(f"WARNING: State status not OK ({new_state.status}) in game {game}. Details logged to {log_file}. Skipping game.")
                        break # Skip to next game

                state = new_state

            # Only count completed games
            if state.final_state:
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                completed_games += 1

        except Exception as e:
            if STRICT_CHECKING:
                raise  # Re-raise the exception in strict mode
            else:
                print(f"Error during evaluation game {game}: {e}")
                # Continue with next game in non-strict mode

    # Return average profit only for completed games
    if completed_games == 0:
        print("WARNING: No games completed during evaluation!")
        return 0.0
    
    avg_profit = total_profit / completed_games
    print(f"Evaluation completed. Completed games: {completed_games}")
    print(f"Total profit: {total_profit:.2f}")
    print(f"Average profit per game: {avg_profit:.2f}")
    return avg_profit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Deep CFR agent checkpoint.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the agent checkpoint file to evaluate.')
    parser.add_argument('--num-games', type=int, default=1000, help='Number of games to run for evaluation.')
    parser.add_argument('--num-players', type=int, default=6, help='Number of players in the game.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output (currently minimal effect).')
    parser.add_argument('--strict', action='store_true', help='Enable strict error checking during evaluation.')

    args = parser.parse_args()

    set_verbose(args.verbose)
    set_strict_checking(args.strict)

    evaluate_model_against_random(
        agent_checkpoint_path=args.checkpoint,
        num_games=args.num_games,
        num_players=args.num_players
    )

    print("\nEvaluation finished.") 