import argparse
import numpy as np
from functions import prepare_stock_data, plot_training_results, calculate_buy_hold_performance
from dqn_model import create_dqn_model, train_dqn
import os
from datetime import datetime
import tensorflow as tf

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train or test DQN model on stock data.")
    
    parser.add_argument("-sd", "--start_date", type=str, required=True, help="Start date in 'YYYY-MM-DD' format")
    parser.add_argument("-ed", "--end_date", type=str, required=True, help="End date in 'YYYY-MM-DD' format")
    parser.add_argument("-m", "--model_path", type=str, help="Path to saved model to load and test (optional)")
    
    return parser.parse_args()

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

def main():
    args = parse_arguments()
    csv_path = "dataset/individual_companies/AAPL_data.csv"
    df_raw, df_state, state_columns, scaler = prepare_stock_data(csv_path, start_date=args.start_date, end_date=args.end_date)

    state_size = len(state_columns)  
    action_size = 3  # Buy, Sell, Hold
    
    if args.model_path:
        # Load and test existing model
        print(f"\nLoading model from {args.model_path}...")
        # Create a new model with the same architecture
        model = create_dqn_model(state_size, action_size)
        # Load the weights from the saved model
        try:
            model.load_weights(args.model_path)
            print("Model loaded successfully!")
        except (ValueError, FileNotFoundError) as e:
            # If the original path fails, try adding or removing .weights in the path
            if ".weights.h5" in args.model_path:
                alternative_path = args.model_path.replace(".weights.h5", ".h5")
            else:
                alternative_path = args.model_path.replace(".h5", ".weights.h5")
            
            print(f"Could not load model with original path. Trying alternative path: {alternative_path}")
            model.load_weights(alternative_path)
            print("Model loaded successfully with alternative path!")
        
        # Calculate benchmark performance
        benchmark_rewards = calculate_buy_hold_performance(df_raw)
        
        # Test model predictions
        test_model_predictions(model, df_state)
        
        # Plot performance
        # Note: For loaded models, we'll need to run an episode to get performance data
        print("\nRunning a test episode to evaluate performance...")
        episode_rewards, epsilons, action_counts, performance_history = train_dqn(
            model, episodes=1, epsilon=0.01, gamma=0.95, 
            epsilon_min=0.01, epsilon_decay=0.85, df=df_state
        )
        
        # Plot results with benchmark
        plot_training_results(episode_rewards, epsilons, action_counts, performance_history, benchmark_rewards)
        
    else:
        # Train new model
        print("\nCreating new DQN model...")
        model = create_dqn_model(state_size, action_size)
        
        episodes = 50
        epsilon = 0.9
        gamma = 0.98
        epsilon_min = 0.01
        epsilon_decay = 0.9

        print("Starting DQN training with epsilon-greedy policy...")
        episode_rewards, epsilons, action_counts, performance_history = train_dqn(
            model, episodes, epsilon, gamma, epsilon_min, epsilon_decay, df_state
        )

        # Calculate benchmark performance
        benchmark_rewards = calculate_buy_hold_performance(df_raw)
        
        # Plot training results with benchmark comparison
        plot_training_results(episode_rewards, epsilons, action_counts, performance_history, benchmark_rewards)

        # Save the model with a timestamp
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_save_path = os.path.join(save_dir, f"dqn_trained_model_{timestamp}.h5")
        
        # Try to save with original extension first, if it fails use the explicit weights extension
        try:
            # Save only the weights instead of the full model
            model.save_weights(model_save_path)
            print(f"\nModel weights saved to {model_save_path}")
        except ValueError as e:
            # If original extension fails, try with explicit weights extension
            model_save_path = os.path.join(save_dir, f"dqn_trained_model_{timestamp}.weights.h5")
            model.save_weights(model_save_path)
            print(f"\nModel weights saved to {model_save_path}")

        # Test model predictions
        test_model_predictions(model, df_state)

if __name__ == "__main__":
    main()