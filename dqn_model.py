import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import pandas as pd
from functions import prepare_stock_data

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

def train_dqn(model, episodes, epsilon, gamma, epsilon_min, epsilon_decay, df):
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

    #total steps: 1239
    #i will add a max_no_steps

    max_no_steps=500
    for e in range(episodes):

        # Reset state
        state_index=0 #first day
        state =  df.iloc[state_index].values.reshape(1, -1) #first real state
        total_reward = 0
        steps=0
        
        done = False
        while not done and state_index<len(df)-1:
            print(f"Step {steps}, State Index: {state_index}/{len(df)}")

            steps+=1
            # Choose action using epsilon greedy policy
            action = epsilon_greedy_policy(model, state, epsilon)
            
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
                reward = (next_price - current_price) * 0.1  # No action so no reward
            
            # Update the value Q (DQN using Bellman Equation)
            q_values = model.predict(state)
            next_q_values = model.predict(next_state)
            
            target = reward + gamma * np.max(next_q_values)
            q_values[0][action] = target
            
            # Train the model using the updated Q value
            if steps % 10 == 0:
                model.fit(state, q_values, verbose=0)
            
            state = next_state
            state_index+=1
            total_reward += reward
            
            # Condition for stopping
            # Stop if we reach the end of the dataset
            if state_index >= max_no_steps:
                done = True
        
        # Reduce the value of epsilon after each episode for more exploitation\
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Steps: {steps}")
   
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
