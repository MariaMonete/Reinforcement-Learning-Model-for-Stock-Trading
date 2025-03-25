import numpy as np
from functions import prepare_stock_data
from dqn_model import create_dqn_model, train_dqn

csv_path = "dataset/individual_companies/AAPL_data.csv"
df_raw, df_state, state_columns, scaler = prepare_stock_data(csv_path)

state_size = len(state_columns)  
action_size = 3  # Buy, Sell, Hold
model = create_dqn_model(state_size, action_size)

episodes = 10  # Before was 100
epsilon = 0.9
gamma = 0.95 
epsilon_min = 0.01
epsilon_decay = 0.995

print("Starting DQN training with epsilon-greedy policy...")
train_dqn(model, episodes, epsilon, gamma, epsilon_min, epsilon_decay, df_state)

# Test trained model
print("\nTesting trained model on a sample state...")
sample_state = np.array(df_state.iloc[0]).reshape(1, -1)  
q_values = model.predict(sample_state)
action = np.argmax(q_values)

# Print results
print("Q-values for sample state:", q_values)
print("Selected action:", action, "(0: Buy, 1: Sell, 2: Hold)")

