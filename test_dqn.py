import argparse
import numpy as np
from functions import prepare_stock_data
from dqn_model import create_dqn_model, train_dqn
import os
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train DQN model on stock data.")
    
    parser.add_argument("-sd", "--start_date", type=str, required=True, help="Start date in 'YYYY-MM-DD' format")
    parser.add_argument("-ed", "--end_date", type=str, required=True, help="End date in 'YYYY-MM-DD' format")
    
    return parser.parse_args()

args = parse_arguments()

csv_path = "dataset/individual_companies/AAPL_data.csv"
df_raw, df_state, state_columns, scaler = prepare_stock_data(csv_path, start_date=args.start_date, end_date=args.end_date)

state_size = len(state_columns)  
action_size = 3  # Buy, Sell, Hold
model = create_dqn_model(state_size, action_size)

episodes = 5
epsilon = 0.9
gamma = 0.95 
epsilon_min = 0.01
epsilon_decay = 0.7

print("Starting DQN training with epsilon-greedy policy...")
train_dqn(model, episodes, epsilon, gamma, epsilon_min, epsilon_decay, df_state)

# Save the model with a timestamp
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
model_save_path = os.path.join(save_dir, f"dqn_trained_model_{timestamp}.h5")
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")

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