# deep_cfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers as pkrs
from collections import deque
from src.core.model import PokerNetwork, encode_state, VERBOSE, set_verbose
from src.utils.settings import STRICT_CHECKING
from src.utils.logging import log_game_error

class PrioritizedMemory:
    """Enhanced memory buffer with prioritized experience replay."""
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize memory buffer with prioritized experience replay.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Controls how much prioritization is used (0 = no prioritization, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self._max_priority = 1.0  # Initial max priority for new experiences
        
    def add(self, experience, priority=None):
        """
        Add a new experience to memory with its priority.
        
        Args:
            experience: Tuple of (state, opponent_features, action_type, bet_size, regret)
            priority: Optional explicit priority value (defaults to max priority if None)
        """
        if priority is None:
            priority = self._max_priority
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.alpha)
        else:
            # Replace the oldest entry
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.alpha
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Controls importance sampling correction (0 = no correction, 1 = full correction)
                 Should be annealed from ~0.4 to 1 during training
                 
        Returns:
            Tuple of (samples, indices, importance_sampling_weights)
        """
        if len(self.buffer) < batch_size:
            # If we don't have enough samples, return all with equal weights
            return self.buffer, list(range(len(self.buffer))), np.ones(len(self.buffer))
        
        # Convert priorities to probabilities
        total_priority = sum(self.priorities)
        probabilities = [p / total_priority for p in self.priorities]
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = []
        for idx in indices:
            # P(i) = p_i^α / sum_k p_k^α
            # weight = (1/N * 1/P(i))^β = (N*P(i))^-β
            sample_prob = self.priorities[idx] / total_priority
            weight = (len(self.buffer) * sample_prob) ** -beta
            weights.append(weight)
        
        # Normalize weights to have maximum weight = 1
        # This ensures we only scale down updates, never up
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        
        return samples, indices, np.array(weights, dtype=np.float32)
        
    def update_priority(self, index, priority):
        """
        Update the priority of an experience.
        
        Args:
            index: Index of the experience to update
            priority: New priority value (before alpha adjustment)
        """
        # Clip priority to be positive
        priority = max(1e-8, priority)
        
        # Keep track of max priority for new experience initialization
        self._max_priority = max(self._max_priority, priority)
        
        # Store alpha-adjusted priority
        self.priorities[index] = priority ** self.alpha
        
    def __len__(self):
        """Return the current size of the memory."""
        return len(self.buffer)
        
    def get_memory_stats(self):
        """Get statistics about the current memory buffer."""
        if not self.priorities:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "size": 0}
            
        raw_priorities = [p ** (1/self.alpha) for p in self.priorities]
        return {
            "min": min(raw_priorities),
            "max": max(raw_priorities),
            "mean": sum(raw_priorities) / len(raw_priorities),
            "median": sorted(raw_priorities)[len(raw_priorities) // 2],
            "size": len(self.buffer)
        }

class DeepCFRAgent:
    def __init__(self, player_id=0, num_players=6, memory_size=300000, device='cpu'):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        
        # Define action types (Fold, Check/Call, Raise)
        self.num_actions = 3
        
        # Calculate input size based on state encoding
        input_size = 52 + 52 + 5 + 1 + num_players + num_players + num_players*4 + 1 + 4 + 5
        
        # Create advantage network with bet sizing
        self.advantage_net = PokerNetwork(input_size=input_size, hidden_size=256, num_actions=self.num_actions).to(device)
        
        # Use a smaller learning rate for more stable training
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=0.00005, weight_decay=1e-5)
        
        # Create prioritized memory buffer
        self.advantage_memory = PrioritizedMemory(memory_size)
        
        # Strategy network
        self.strategy_net = PokerNetwork(input_size=input_size, hidden_size=256, num_actions=self.num_actions).to(device)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.00005, weight_decay=1e-5)
        self.strategy_memory = deque(maxlen=memory_size)
        
        # For keeping statistics
        self.iteration_count = 0
        
        # Regret normalization tracker
        self.max_regret_seen = 1.0
        
        # Bet sizing bounds (as multipliers of pot)
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0

    def action_type_to_pokers_action(self, action_type, state, bet_size_multiplier=None):
        """
        Convert action type and optional bet size to Pokers action.

        Args:
            action_type: Integer action type (0=fold, 1=check/call, 2=raise)
            state: Current poker game state
            bet_size_multiplier: Multiplier for pot-sized bets (typically 0.1-3x pot)

        Returns:
            A valid pokers.Action object
        """
        try:
            if action_type == 0:  # Fold
                # Ensure Fold is legal before returning
                if pkrs.ActionEnum.Fold in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                else:
                    # Fallback if Fold is somehow illegal (shouldn't happen often)
                    if pkrs.ActionEnum.Check in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    elif pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    else:
                        # If nothing else is legal, something is very wrong
                        print("WARNING: No legal actions found, even Fold!")
                        # Attempt Fold anyway as a last resort
                        return pkrs.Action(pkrs.ActionEnum.Fold)

            elif action_type == 1:  # Check/Call
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                elif pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                else:
                    # Fallback if neither Check nor Call is legal
                    if pkrs.ActionEnum.Fold in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Fold)
                    else:
                        print("WARNING: Check/Call chosen but neither is legal!")
                        # Attempt Check as a last resort if available
                        return pkrs.Action(pkrs.ActionEnum.Check)


            elif action_type == 2:  # Raise
                # First, check if Raise itself is a legal action type
                if pkrs.ActionEnum.Raise not in state.legal_actions:
                    # If Raise is not legal, fall back to Call or Check
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    elif pkrs.ActionEnum.Check in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    else:
                        # If neither Call nor Check is available, Fold
                        return pkrs.Action(pkrs.ActionEnum.Fold)

                # Get current player state
                player_state = state.players_state[state.current_player]
                current_bet = player_state.bet_chips
                available_stake = player_state.stake

                # Calculate what's needed to call (match the current min_bet)
                call_amount = max(0, state.min_bet - current_bet)

                # *** CORRECTED LOGIC HERE ***
                # If player doesn't have enough chips *beyond* the call amount to make a valid raise,
                # or if they are going all-in just to call, it should be a Call action.
                # A raise requires putting in *more* than the call amount.
                # The minimum additional raise amount is typically the big blind or 1 chip.
                min_raise_increment = 1.0 # A small default
                if hasattr(state, 'bb'):
                    min_raise_increment = max(1.0, state.bb) # Usually BB, but at least 1

                if available_stake <= call_amount + min_raise_increment:
                    # Player cannot make a valid raise (or is all-in just to call).
                    # This action should be treated as a Call.
                    if VERBOSE:
                        print(f"Action type 2 (Raise) chosen, but player cannot make a valid raise. "
                              f"Stake: {available_stake}, Call Amount: {call_amount}. Switching to Call.")
                    # Ensure Call is legal before returning it
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    else:
                        # If Call is not legal (edge case, e.g., already all-in matching bet), Fold.
                         if VERBOSE:
                             print(f"WARNING: Cannot Call (not legal), falling back to Fold.")
                         return pkrs.Action(pkrs.ActionEnum.Fold)
                # *** END OF CORRECTION ***

                # If we reach here, the player *can* make a valid raise.
                remaining_stake_after_call = available_stake - call_amount

                # Calculate target raise amount based on pot multiplier
                pot_size = max(1.0, state.pot) # Avoid division by zero

                # Apply dynamic bet sizing if appropriate
                if bet_size_multiplier is None:
                    # Default to 1x pot if no multiplier provided
                    bet_size_multiplier = 1.0
                else:
                    # Adjust bet size based on game state
                    bet_size_multiplier = self.adjust_bet_size(state, bet_size_multiplier)

                # Ensure multiplier is within bounds
                bet_size_multiplier = max(self.min_bet_size, min(self.max_bet_size, bet_size_multiplier))
                target_additional_raise = pot_size * bet_size_multiplier

                # Ensure minimum raise increment is met
                target_additional_raise = max(target_additional_raise, min_raise_increment)

                # Ensure we don't exceed available stake after calling
                additional_amount = min(target_additional_raise, remaining_stake_after_call)

                # Final check: Ensure the additional amount is at least the minimum required increment
                if additional_amount < min_raise_increment:
                     # This case should be rare due to the check above, but as a safeguard:
                     if VERBOSE:
                         print(f"Calculated raise amount {additional_amount} is less than min increment {min_raise_increment}. Falling back to Call.")
                     if pkrs.ActionEnum.Call in state.legal_actions:
                         return pkrs.Action(pkrs.ActionEnum.Call)
                     else:
                         return pkrs.Action(pkrs.ActionEnum.Fold)


                if VERBOSE:
                    print(f"\nRAISE CALCULATION DETAILS:")
                    print(f"  Player ID: {state.current_player}")
                    print(f"  Action type: {action_type}")
                    print(f"  Current bet: {current_bet}")
                    print(f"  Available stake: {available_stake}")
                    print(f"  Min bet: {state.min_bet}")
                    print(f"  Call amount: {call_amount}")
                    print(f"  Pot size: {state.pot}")
                    print(f"  Bet multiplier: {bet_size_multiplier}x pot")
                    print(f"  Calculated additional raise amount: {additional_amount}")
                    print(f"  Total player bet will be: {current_bet + call_amount + additional_amount}")

                # Return the Raise action with the calculated *additional* amount
                return pkrs.Action(pkrs.ActionEnum.Raise, additional_amount)

            else:
                raise ValueError(f"Unknown action type: {action_type}")

        except Exception as e:
            if VERBOSE:
                print(f"ERROR creating action {action_type}: {e}")
                print(f"State: current_player={state.current_player}, legal_actions={state.legal_actions}")
                print(f"Player stake: {state.players_state[state.current_player].stake}")

            # Fall back to a safe action
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            elif pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            else:
                return pkrs.Action(pkrs.ActionEnum.Fold)

    def adjust_bet_size(self, state, base_multiplier):
        """
        Dynamically adjust bet size multiplier based on game state.
        
        Args:
            state: Current poker game state
            base_multiplier: Base bet size multiplier from the model
            
        Returns:
            Adjusted bet size multiplier
        """
        # Default adjustment factor
        adjustment = 1.0
        
        # Adjust based on game stage
        if int(state.stage) >= 2:  # Turn or River
            adjustment *= 1.2  # Increase bets in later streets
        
        # Adjust based on pot size relative to starting stack
        initial_stake = state.players_state[0].stake + state.players_state[0].bet_chips
        pot_ratio = state.pot / initial_stake
        if pot_ratio > 0.5:  # Large pot
            adjustment *= 1.1  # Bet bigger in large pots
        elif pot_ratio < 0.1:  # Small pot
            adjustment *= 0.9  # Bet smaller in small pots
        
        # Adjust based on position (more aggressive in late position)
        btn_distance = (state.current_player - state.button) % len(state.players_state)
        if btn_distance <= 1:  # Button or cutoff
            adjustment *= 1.15  # More aggressive in late position
        elif btn_distance >= 4:  # Early position
            adjustment *= 0.9  # Less aggressive in early position
        
        # Adjust for number of active players (larger with fewer players)
        active_players = sum(1 for p in state.players_state if p.active)
        if active_players <= 2:
            adjustment *= 1.2  # Larger bets heads-up
        elif active_players >= 5:
            adjustment *= 0.9  # Smaller bets multiway
        
        # Apply adjustment to base multiplier
        adjusted_multiplier = base_multiplier * adjustment
        
        # Ensure we stay within bounds
        return max(self.min_bet_size, min(self.max_bet_size, adjusted_multiplier))

    def get_legal_action_types(self, state):
        """Get the legal action types for the current state."""
        legal_action_types = []
        
        # Check each action type
        if pkrs.ActionEnum.Fold in state.legal_actions:
            legal_action_types.append(0)
            
        if pkrs.ActionEnum.Check in state.legal_actions or pkrs.ActionEnum.Call in state.legal_actions:
            legal_action_types.append(1)
            
        if pkrs.ActionEnum.Raise in state.legal_actions:
            legal_action_types.append(2)
        
        return legal_action_types

    def cfr_traverse(self, state, iteration, random_agents, depth=0):
        """
        Traverse the game tree using external sampling MCCFR with continuous bet sizing.
        
        Args:
            state: Current game state
            iteration: Current training iteration
            random_agents: List of opponent agents
            depth: Current recursion depth
            
        Returns:
            Expected value for the current player
        """
        # Add recursion depth protection
        max_depth = 1000
        if depth > max_depth:
            if VERBOSE:
                print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
            return 0
        
        if state.final_state:
            # Return payoff for the trained agent
            return state.players_state[self.player_id].reward
        
        current_player = state.current_player
        
        # If it's the trained agent's turn
        if current_player == self.player_id:
            legal_action_types = self.get_legal_action_types(state)
            
            if not legal_action_types:
                if VERBOSE:
                    print(f"WARNING: No legal actions found for player {current_player} at depth {depth}")
                return 0
                
            # Encode the base state
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(self.device)
            
            # Get advantages and bet sizing prediction from network
            with torch.no_grad():
                advantages, bet_size_pred = self.advantage_net(state_tensor.unsqueeze(0))
                advantages = advantages[0].cpu().numpy()
                bet_size_multiplier = bet_size_pred[0][0].item()
            
            # Use regret matching to compute strategy for action types
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_types:
                advantages_masked[a] = max(advantages[a], 0)
                
            # Choose an action based on the strategy
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_types:
                    strategy[a] = 1.0 / len(legal_action_types)
            
            # Choose actions and traverse
            action_values = np.zeros(self.num_actions)
            for action_type in legal_action_types:
                try:
                    # Use the predicted bet size for raise actions
                    if action_type == 2:  # Raise
                        pokers_action = self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
                    else:
                        pokers_action = self.action_type_to_pokers_action(action_type, state)
                    
                    new_state = state.apply_action(pokers_action)
                    
                    # Check if the action was valid
                    if new_state.status != pkrs.StateStatus.Ok:
                        log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                        if STRICT_CHECKING:
                            raise ValueError(f"State status not OK ({new_state.status}) during CFR traversal. Details logged to {log_file}")
                        elif VERBOSE:
                            print(f"WARNING: Invalid action {action_type} at depth {depth}. Status: {new_state.status}")
                            print(f"Player: {current_player}, Action: {pokers_action.action}, Amount: {pokers_action.amount if pokers_action.action == pkrs.ActionEnum.Raise else 'N/A'}")
                            print(f"Current bet: {state.players_state[current_player].bet_chips}, Stake: {state.players_state[current_player].stake}")
                            print(f"Details logged to {log_file}")
                        continue  # Skip this action and try others in non-strict mode
                        
                    action_values[action_type] = self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
                except Exception as e:
                    if VERBOSE:
                        print(f"ERROR in traversal for action {action_type}: {e}")
                    action_values[action_type] = 0
                    if STRICT_CHECKING:
                        raise  # Re-raise in strict mode
            
            # Compute counterfactual regrets and add to memory
            ev = sum(strategy[a] * action_values[a] for a in legal_action_types)
            
            # Calculate normalization factor
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            
            for action_type in legal_action_types:
                # Calculate regret
                regret = action_values[action_type] - ev
                
                # Normalize and clip regret
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
                
                # Apply scaling
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0  # Linear CFR
                weighted_regret = clipped_regret * scale_factor
                
                # Store in prioritized memory with regret magnitude as priority
                priority = abs(weighted_regret) + 0.01  # Add small constant to ensure non-zero priority
                
                # For raise actions, store the bet size multiplier
                if action_type == 2:
                    self.advantage_memory.add(
                        (encode_state(state, self.player_id), 
                         np.zeros(20),  # placeholder for opponent features 
                         action_type, 
                         bet_size_multiplier, 
                         weighted_regret),
                        priority
                    )
                else:
                    self.advantage_memory.add(
                        (encode_state(state, self.player_id),
                         np.zeros(20),  # placeholder for opponent features
                         action_type, 
                         0.0,  # Default bet size for non-raise actions 
                         weighted_regret),
                        priority
                    )
            
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_types:
                strategy_full[a] = strategy[a]
            
            self.strategy_memory.append((
                encode_state(state, self.player_id),
                np.zeros(20),  # placeholder for opponent features
                strategy_full,
                bet_size_multiplier if 2 in legal_action_types else 0.0,
                iteration
            ))
            
            return ev
            
        # If it's another player's turn (random agent)
        else:
            try:
                # Let the random agent choose an action
                action = random_agents[current_player].choose_action(state)
                new_state = state.apply_action(action)
                
                # Check if the action was valid
                if new_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                    if STRICT_CHECKING:
                        raise ValueError(f"State status not OK ({new_state.status}) from random agent. Details logged to {log_file}")
                    if VERBOSE:
                        print(f"WARNING: Random agent made invalid action at depth {depth}. Status: {new_state.status}")
                        print(f"Details logged to {log_file}")
                    return 0
                    
                return self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
            except Exception as e:
                if VERBOSE:
                    print(f"ERROR in random agent traversal: {e}")
                if STRICT_CHECKING:
                    raise  # Re-raise in strict mode
                return 0

    def train_advantage_network(self, batch_size=128, epochs=3, beta_start=0.4, beta_end=1.0):
        """
        Train the advantage network using prioritized experience replay.
        """
        if len(self.advantage_memory) < batch_size:
            return 0
        
        self.advantage_net.train()
        total_loss = 0
        
        # Calculate current beta for importance sampling
        progress = min(1.0, self.iteration_count / 10000)
        beta = beta_start + progress * (beta_end - beta_start)
        
        for epoch in range(epochs):
            # Sample batch from prioritized memory with current beta
            batch, indices, weights = self.advantage_memory.sample(batch_size, beta=beta)
            states, opponent_features, action_types, bet_sizes, regrets = zip(*batch)
            
            # Convert to tensors
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            action_type_tensors = torch.LongTensor(np.array(action_types)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            weight_tensors = torch.FloatTensor(weights).to(self.device)
            
            # Forward pass
            action_advantages, bet_size_preds = self.advantage_net(state_tensors, opponent_feature_tensors)
            
            # Compute action type loss (for all actions)
            predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
            action_loss = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
            weighted_action_loss = (action_loss * weight_tensors).mean()
            
            # Compute bet sizing loss (only for raise actions)
            raise_mask = (action_type_tensors == 2)
            if torch.any(raise_mask):
                # Calculate loss for all bet sizes
                all_bet_losses = F.smooth_l1_loss(bet_size_preds, bet_size_tensors, reduction='none')
                
                # Only count losses for raise actions, zero out others
                masked_bet_losses = all_bet_losses * raise_mask.float().unsqueeze(1)
                
                # Calculate weighted average loss
                raise_count = raise_mask.sum().item()
                if raise_count > 0:
                    weighted_bet_size_loss = (masked_bet_losses.squeeze() * weight_tensors).sum() / raise_count
                    combined_loss = weighted_action_loss + 0.5 * weighted_bet_size_loss
                else:
                    combined_loss = weighted_action_loss
            else:
                combined_loss = weighted_action_loss
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            combined_loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Update priorities based on new losses
            with torch.no_grad():
                # Calculate new errors for priority updates
                new_action_errors = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
                
                # For raise actions, include bet sizing errors
                if torch.any(raise_mask):
                    # Calculate normalized bet size errors for each sample
                    new_bet_errors = torch.zeros_like(new_action_errors)
                    
                    # Only add bet sizing errors for raise actions
                    raise_indices = torch.where(raise_mask)[0]
                    for i in raise_indices:
                        new_bet_errors[i] = F.smooth_l1_loss(
                            bet_size_preds[i], bet_size_tensors[i], reduction='mean'
                        )
                    
                    # Combined error with smaller weight for bet sizing
                    combined_errors = new_action_errors + 0.5 * new_bet_errors
                else:
                    combined_errors = new_action_errors
                
                # Update priorities
                combined_errors_np = combined_errors.cpu().numpy()
                for i, idx in enumerate(indices):
                    self.advantage_memory.update_priority(idx, combined_errors_np[i] + 0.01)
            
            total_loss += combined_loss.item()
        
        # Return average loss
        return total_loss / epochs

    def train_strategy_network(self, batch_size=128, epochs=3):
        """
        Train the strategy network using collected samples.
        
        Args:
            batch_size: Size of training batches
            epochs: Number of training epochs per call
            
        Returns:
            Average training loss
        """
        if len(self.strategy_memory) < batch_size:
            return 0
        
        self.strategy_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch from memory
            batch = random.sample(self.strategy_memory, batch_size)
            states, opponent_features, strategies, bet_sizes, iterations = zip(*batch)
            
            # Convert to tensors
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            
            # Weight samples by iteration (Linear CFR)
            weights = iteration_tensors / torch.sum(iteration_tensors)
            
            # Forward pass
            action_logits, bet_size_preds = self.strategy_net(state_tensors)
            predicted_strategies = F.softmax(action_logits, dim=1)
            
            # Action type loss (weighted cross-entropy)
            # Add small epsilon to prevent log(0)
            action_loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1))
            
            # Bet size loss (only for states with raise actions)
            raise_mask = (strategy_tensors[:, 2] > 0)
            if raise_mask.sum() > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_preds = bet_size_preds[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weights[raise_indices]
                
                # Use huber loss for bet sizing to be more robust to outliers
                bet_size_loss = F.smooth_l1_loss(raise_bet_preds, raise_bet_targets, reduction='none')
                weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss.squeeze())
                
                # Combine losses with appropriate weighting
                combined_loss = action_loss + 0.5 * weighted_bet_size_loss
            else:
                combined_loss = action_loss
            
            # Backward pass and optimize
            self.strategy_optimizer.zero_grad()
            combined_loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=0.5)
            
            self.strategy_optimizer.step()
            
            total_loss += combined_loss.item()
        
        # Return average loss
        return total_loss / epochs

    def choose_action(self, state):
        """Choose an action for the given state during actual play."""
        legal_action_types = self.get_legal_action_types(state)
        
        if not legal_action_types:
            # Default to call if no legal actions (shouldn't happen)
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            elif pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            else:
                return pkrs.Action(pkrs.ActionEnum.Fold)
            
        state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, bet_size_pred = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()
        
        # Filter to only legal actions
        legal_probs = np.array([probs[a] for a in legal_action_types])
        if np.sum(legal_probs) > 0:
            legal_probs = legal_probs / np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)
        
        # Choose action based on probabilities
        action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
        action_type = legal_action_types[action_idx]
        
        # Use the predicted bet size for raise actions
        if action_type == 2:  # Raise
            return self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
        else:
            return self.action_type_to_pokers_action(action_type, state)

    def save_model(self, path_prefix):
        """Save the model to disk."""
        torch.save({
            'iteration': self.iteration_count,
            'advantage_net': self.advantage_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict(),
            'min_bet_size': self.min_bet_size,
            'max_bet_size': self.max_bet_size
        }, f"{path_prefix}_iteration_{self.iteration_count}.pt")
        
    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.iteration_count = checkpoint['iteration']
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
        
        # Load bet size bounds if available in the checkpoint
        if 'min_bet_size' in checkpoint:
            self.min_bet_size = checkpoint['min_bet_size']
        if 'max_bet_size' in checkpoint:
            self.max_bet_size = checkpoint['max_bet_size']