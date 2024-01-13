import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from talib import RSI, MACD

def create_technical_indicators(data):
    data['RSI'] = RSI(data['Close'])
    data['MACD'], _, _ = MACD(data['Close'])
    return data

def create_features(data):
    data['RSI_Signal'] = np.where(data['RSI'] > 70, -1, 0)
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, data['RSI_Signal'])
    data['MACD_Signal'] = np.where(data['MACD'] > 0, 1, -1)
    return data

def prepare_data(data):
    data = create_technical_indicators(data)
    data = create_features(data)
    data = data.dropna()
    features = ['RSI', 'MACD']
    X = data[features]
    y = data['RSI_Signal']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    return model

def backtest_strategy(data, model):
    data = create_technical_indicators(data)
    data = create_features(data)
    data = data.dropna()
    data['Predicted_Signal'] = model.predict(data[['RSI', 'MACD']])
    data['Actual_Signal'] = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
    return data

def simulate_trading(data):
    capital = 100000 
    position = 0 
    for index, row in data.iterrows():
        if row['Predicted_Signal'] == 1:
            position = capital / row['Close']
            capital = 0
        elif row['Predicted_Signal'] == -1:
            capital = position * row['Close']
            position = 0
    return capital + position * data.iloc[-1]['Close']

if __name__ == "__main__":
    stock_data = pd.read_csv('historical_stock_data.csv')
    
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    backtest_data = backtest_strategy(stock_data.copy(), model)
    final_portfolio_value = simulate_trading(backtest_data)
    
    print(f"Final Portfolio Value: {final_portfolio_value}")
