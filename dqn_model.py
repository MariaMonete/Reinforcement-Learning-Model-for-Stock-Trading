from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

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