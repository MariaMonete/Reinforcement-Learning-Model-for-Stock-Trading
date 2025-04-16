import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import pandas as pd
from functions import prepare_stock_data, plot_training_results
from relay_buffer import ReplayBuffer


def create_dqn_model(state_size, action_size, learning_rate=3e-5):
    """
    Create and compile a Deep Q-Network (DQN) model.
    
    Parameters:
    state_size (int): Number of features in the state representation.
    action_size (int): Number of possible actions.
    learning_rate (float): Learning rate for the Adam optimizer.
    
    Returns:
    model (tf.keras.Model): Compiled DQN model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(state_size,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

def create_target_network(model):
    """
    Create a target network as a copy of the main DQN model.
    
    Parameters:
    model (tf.keras.Model): The main DQN model to copy.
    
    Returns:
    target_model (tf.keras.Model): A copy of the main model with identical weights.
    """
    target_model = Sequential([
        Dense(64, activation='relu', input_shape=(model.input_shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(model.output_shape[1], activation='linear')
    ])
    
    # Copy the weights from the main model
    target_model.set_weights(model.get_weights())
    
    return target_model

def update_target_network(model, target_model, tau=1.0):
    """
    Update the target network weights.
    
    Parameters:
    model (tf.keras.Model): The main DQN model.
    target_model (tf.keras.Model): The target network.
    tau (float): Update rate (1.0 for hard update, <1.0 for soft update).
    """
    if tau == 1.0:
        # Hard update - direct copy
        target_model.set_weights(model.get_weights())
    else:
        # Soft update - gradual blending
        target_weights = target_model.get_weights()
        main_weights = model.get_weights()
        
        updated_weights = []
        for main_w, target_w in zip(main_weights, target_weights):
            updated_w = tau * main_w + (1 - tau) * target_w
            updated_weights.append(updated_w)

        target_model.set_weights(updated_weights)

def epsilon_greedy_policy(model, state, epsilon, position=None):
    """
    Epsilon-Greedy policy with position-based action masking
    
    Parameters:
    model (tf.keras.Model): DQN model.
    state (np.ndarray): Current state.
    epsilon (float): Probability of exploration.
    position (int, optional): Current position (1=long, -1=short, 0=no position).
    
    Returns:
    action (int): Index of chosen action.
    """
    if np.random.rand() <= epsilon:
        # Exploration: random choice with position constraints
        if position is not None:
            if position == 1:  # Already in long position
                valid_actions = [1, 2]  # Can only sell or hold
            elif position == -1:  # Already in short position
                valid_actions = [0, 2]  # Can only buy or hold
            else:  # No position
                valid_actions = [0, 1, 2]  # Can do any action
            action = np.random.choice(valid_actions)
        else:
            # If position is not provided, use regular random selection
            action = np.random.choice(model.output_shape[1])
    else:
        # Exploitation: action with biggest Q value, with position constraints
        q_values = model.predict(state)
        
        if position is not None:
            # Create a mask based on position
            mask = np.ones(model.output_shape[1])
            if position == 1:  # Already long
                mask[0] = -np.inf  # Mask buy action
            elif position == -1:  # Already short
                mask[1] = -np.inf  # Mask sell action
            
            # Apply mask to q_values
            masked_q_values = q_values + mask
            action = np.argmax(masked_q_values)
        else:
            action = np.argmax(q_values)
    
    return action

df = prepare_stock_data("dataset/individual_companies/AAPL_data.csv")

def train_dqn(model, episodes, epsilon, gamma, epsilon_min, epsilon_decay, df, batch_size=32, buffer_size=10000, alpha=0.6, beta_start=0.4, beta_end=1.0):
    """
    Train the agent using Q-learning and epsilon-greedy policy

    Parameters:
    model (tf.keras.Model): DQN model
    episodes (int): Number of episodes for training
    epsilon (float): Initial value of epsilon
    gamma (float): Discount factor
    epsilon_min (float): Min value of epsilon
    epsilon_decay (float): Decay rate of epsilon
    df (pd.DataFrame): Stock market data
    
    Returns:
    tuple: (episode_rewards, epsilons, action_counts, performance_history)
    """
    target_model = create_target_network(model)

    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        alpha=alpha,
        beta=beta_start,
        beta_increment=(beta_end - beta_start) / episodes  # Gradually increase beta to 1
    )

    episode_rewards = []
    epsilons = []
    action_counts = [0, 0, 0]  # [Buy, Sell, Hold]
    performance_history = []  # Track performance over time

    for e in range(episodes):
        # Reset state
        state_index = 0  # first day
        state = df.iloc[state_index].values.reshape(1, -1)  # first real state
        total_reward = 0
        steps = 0
        position = 0
        episode_performance = []  # Track performance for this episode
        
        done = False
        while not done and state_index < len(df)-1:
            if steps % 50 == 0:
                print(f"Episode {e+1}, Step {steps}, State Index: {state_index}/{len(df)}")

            steps += 1
            # Choose action using epsilon greedy policy with position constraints
            action = epsilon_greedy_policy(model, state, epsilon, position)
            action_counts[action] += 1
            
            # Execute the action and observe the next state and the reward
            next_index = state_index + 1
            next_state = df.iloc[next_index].values.reshape(1, -1)  # next real state

            current_price = df.iloc[state_index]["return_close"]
            next_price = df.iloc[next_index]["return_close"]

            if action == 0:  # Buy
                if position == 0:  # Only buy if no position
                    position = 1
                    reward = next_price - current_price
                elif position == -1:  # Close short position
                    position = 0
                    reward = (current_price - next_price) * 0.5  # Partial reward for closing position
                else:
                    # Should not happen with constraints, but just in case
                    reward = -0.01  # Larger penalty for invalid action
            elif action == 1:  # Sell
                if position == 0:  # Only sell if no position
                    position = -1
                    reward = current_price - next_price
                elif position == 1:  # Close long position
                    position = 0
                    reward = (next_price - current_price) * 0.5  # Partial reward for closing position
                else:
                    # Should not happen with constraints, but just in case
                    reward = -0.01  # Larger penalty for invalid action
            else:  # Hold
                if position == 0:
                    reward = 0  # No reward/penalty for holding cash
                else:
                    # Reward based on position direction
                    reward = position * (next_price - current_price)

            # Condition for stopping
            # Stop if we reach the end of the dataset
            done = state_index >= len(df)-1

            # Add experience to replay buffer
            replay_buffer.add(state[0], action, reward, next_state[0], done)
            
            state = next_state
            state_index += 1
            total_reward += reward

            # Track performance over time
            episode_performance.append(total_reward)

            if replay_buffer.size() >= batch_size:
                # Sample batch from replay buffer
                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
                
                # Reshape for model prediction
                states = states.reshape(batch_size, -1)
                next_states = next_states.reshape(batch_size, -1)
                
                # Compute target Q-values
                current_q_values = model.predict(states)
                next_q_values = target_model.predict(next_states)

                # Initialize TD errors
                td_errors = np.zeros(batch_size)
                
                # Update Q-values
                for i in range(batch_size):
                    old_val = current_q_values[i][actions[i]]
                    
                    if dones[i]:
                        target = rewards[i]
                    else:
                        target = rewards[i] + gamma * np.max(next_q_values[i])
                    
                    # Store TD error for updating priorities
                    td_errors[i] = abs(target - old_val)
                    
                    # Update Q-value with importance sampling weight
                    current_q_values[i][actions[i]] = old_val + weights[i] * (target - old_val)
                
                # Train on batch
                model.fit(states, current_q_values, verbose=0, batch_size=batch_size, epochs=1)

                # Update priorities in the replay buffer
                replay_buffer.update_priorities(indices, td_errors)
        
            if steps % 100 == 0:
                update_target_network(model, target_model)

        episode_rewards.append(total_reward)
        epsilons.append(epsilon)
        performance_history.append(episode_performance)  # Store episode performance

        # Reduce the value of epsilon after each episode for more exploitation
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Steps: {steps}")

    return episode_rewards, epsilons, action_counts, performance_history