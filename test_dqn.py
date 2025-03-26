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
print("\nTesting trained model on multiple sample states...")
def test_model_predictions(model, df_state, num_samples=5):
    """
    Test the trained model by predicting actions for multiple sample states
    
    Parameters:
    model (tf.keras.Model): Trained DQN model
    df_state (pd.DataFrame): Prepared state data
    num_samples (int): Number of sample states to test
    
    Returns:
    None (prints results)
    """
    print(f"\nTesting model predictions for {num_samples} sample states:")
    for i in range(num_samples):
        # Select a random state from the dataset
        sample_state = np.array(df_state.iloc[i]).reshape(1, -1)
        
        # Predict Q-values
        q_values = model.predict(sample_state)
        action = np.argmax(q_values)
        
        # Interpret the action
        action_map = {0: "Buy", 1: "Sell", 2: "Hold"}
        
        print(f"\nSample State {i+1}:")
        print("Q-values:", q_values[0])
        print("Selected action:", action_map[action])
        print("Raw action index:", action)

# Run the test
test_model_predictions(model, df_state)