import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import pandas as pd
from functions import prepare_stock_data
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
        Dropout(0.2),
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

def epsilon_greedy_policy(model, state, epsilon):
    """
    Epsilon-Greedy policy
    
    Parameters:
    model (tf.keras.Model): DQN model.
    state (np.ndarray): Current state.
    epsilon (float): Probability of exploration.
    
    Returns:
    action (int): Index of chosen action.
    """
    if np.random.rand() <= epsilon:
        # Exploration: random choice
        action = np.random.choice(model.output_shape[1])
    else:
        # Exploitation: action with biggest Q value
        q_values = model.predict(state)
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

    #plot cumulative performance
    cumulative_rewards=[]
    total_cumulative_reward=0

    max_no_steps=500
    for e in range(episodes):

        # Reset state
        state_index=0 #first day
        state =  df.iloc[state_index].values.reshape(1, -1) #first real state
        total_reward = 0
        steps=0
        
        done = False
        while not done and state_index<len(df)-1:
            if steps % 50 == 0:
                print(f"Episode {e+1}, Step {steps}, State Index: {state_index}/{len(df)}")

            steps+=1
            # Choose action using epsilon greedy policy
            action = epsilon_greedy_policy(model, state, epsilon)
            action_counts[action] += 1
            
            # Execute the action and observe the next state and the reward
            next_index=state_index+1
            next_state = df.iloc[next_index].values.reshape(1, -1) #next real state

            current_price=df.iloc[state_index]["return_close"]
            next_price=df.iloc[next_index]["return_close"]

            if action == 0:  # Buy
                reward = next_price - current_price  # Profit/Loss from buying
            elif action == 1:  # Sell
                reward = current_price - next_price  # Profit/Loss from selling
            else:  # Hold
                reward = (next_price - current_price) * 0.1 -0.01  #small penalty to discourage holding

            # Condition for stopping
            # Stop if we reach the end of the dataset
            done = state_index >= len(df)-1

            replay_buffer.add(state[0], action, reward, next_state[0], done)
            
            state = next_state
            state_index+=1
            total_reward += reward

            #update cumulative rewards
            total_cumulative_reward+=total_reward
            cumulative_rewards.append(total_cumulative_reward)

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

        # Reduce the value of epsilon after each episode for more exploitation\
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Steps: {steps}")

    # Plot results after training
    plot_training_results(episode_rewards, epsilons, action_counts, cumulative_rewards)

def plot_training_results(rewards, epsilons, action_counts,cumulative_rewards):
    episodes = range(len(rewards))

    plt.figure(figsize=(12, 4))

    # Reward Plot
    plt.subplot(1, 3, 1)
    plt.plot(episodes, rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()

    # Cumulative Reward Plot
    plt.subplot(1, 4, 2)
    plt.plot(range(len(cumulative_rewards)), cumulative_rewards, label="Cumulative Reward", color="purple")

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Performance Over Time")
    plt.legend()

    # Epsilon Decay
    plt.subplot(1, 3, 2)
    plt.plot(episodes, epsilons, label="Epsilon Decay", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Time")
    plt.legend()

    # Action Distribution
    plt.subplot(1, 3, 3)
    plt.bar(["Buy", "Sell", "Hold"], action_counts, color=["green", "red", "blue"])
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action Selection Distribution")

    plt.tight_layout()
    plt.show()