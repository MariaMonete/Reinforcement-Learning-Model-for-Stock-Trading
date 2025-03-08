# Reinforcement-Learning-Model-for-Stock-Trading

## Project Overview

This team project aims to develop a Deep Q-Network (DQN) reinforcement learning model for automated stock trading using historical S&P 500 data. We're starting with a focused approach and will gradually expand our scope as we validate our methods.

## Dataset

We're using the [S&P 500 dataset from Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500/data), which includes:
- Daily price data (Open, High, Low, Close)
- Trading volume
- Historical data for 500+ companies in the S&P 500 index
- Date range covering 5+ years of market activity

## Project Goals

### Phase 1: Single-Stock Model
- Train a DQN model on historical data for a single company (Apple - AAPL)
- Implement a simple trading environment with basic actions (buy, sell, hold)
- Define a state representation using price and volume data
- Establish baseline performance metrics
- Create visualization tools to understand model decisions

### Phase 2: Model Refinement
- Enhance the state representation with technical indicators (RSI, MACD, Bollinger Bands)
- Improve the reward function to balance returns and risk
- Implement a more realistic simulation with transaction costs
- Test various hyperparameter configurations
- Perform robust backtesting on different time periods

### Phase 3: Multi-Stock Expansion
- Extend the model to handle multiple stocks simultaneously
- Add risk constraints and position sizing
- Test on different market sectors
- Compare performance against benchmark indices

### Phase 4: Advanced Features
- Incorporate market regime detection
- Test transfer learning between different stocks
- Implement adaptive learning rates based on market volatility
- Explore ensemble approaches combining multiple models
- Benchmark against traditional trading strategies

## Model Architecture

Our DQN implementation includes:

- **State Space**: Features derived from price history and technical indicators
- **Action Space**: Discrete actions (buy, sell, hold) with varying position sizes
- **Reward Function**: Risk-adjusted returns (Sharpe ratio) with penalty for excessive trading
- **Network Architecture**: Deep neural network with fully connected layers
- **Experience Replay**: To improve stability and sample efficiency
- **Target Network**: To reduce overestimation of Q-values
- **$\epsilon$-greedy**: To balance exploration-exploitation

## Evaluation Metrics

- Total Return
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown
- Win Rate
- Profit Factor
- Consistency across different market conditions

## Getting Started

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Download the dataset from Kaggle
4. Run data preprocessing: `python preprocess.py`
5. Train the model: `python train.py --stock AAPL --start_date 2010-01-01 --end_date 2015-12-31`
6. Evaluate results: `python evaluate.py --model_path models/aapl_dqn.h5`

## Project Structure

```
reinforcemnet-learning-model-for-stock-trading/
├── dataset/              # Raw and processed data
└── requirements.txt      # Dependencies
```

## Future Directions

- Explore other RL algorithms (PPO, SAC)
- Incorporate sentiment analysis from news data
- Implement online learning for model adaptation
- Develop ensemble strategies combining multiple approaches
- Test on higher frequency data

## Ackwoledgements

This repository was created by Andreea Maria Monete and Vlad Florin Filip