import tensorflow as tf
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

