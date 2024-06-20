import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from models.cnn_lstm_model import build_cnn_lstm
from models.lstnet_model import build_lstnet
from models.tcn_model import build_tcn
from utils.project_functions import load_data, reset_random_seeds
from data.data_preparation import prepare_data, save_preprocessed_data

def train_model(model, X_train, y_train, model_name):
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    model.fit(X_train, y_train, epochs=1000, batch_size=50, validation_split=0.1, callbacks=[early_stopping])
    
    # Ensure the save directory exists
    save_dir = 'model_save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the model architecture and weights
    model.save(os.path.join(save_dir, f'{model_name}.keras'))

def main():
    data_path = 'data/processed/Boruta_data.csv'
    df = load_data(data_path)
    if df is None:
        print("Error loading data.")
        return
    
    timesteps = 5
    X_train, X_test, y_train, y_test, time_test, price, scaler = prepare_data(df, timesteps)
    
    # Save preprocessed data
    save_preprocessed_data('data/processed/preprocessed_data.pkl', X_train, X_test, y_train, y_test, time_test, price, scaler)
    
    # Train CNN-LSTM model
    reset_random_seeds()
    input_shape = (timesteps, X_train.shape[2])
    cnn_lstm_model = build_cnn_lstm(input_shape)
    train_model(cnn_lstm_model, X_train, y_train, 'cnn_lstm')
    
    # Train LSTNet model
    reset_random_seeds()
    lstnet_model = build_lstnet(input_shape)
    train_model(lstnet_model, X_train, y_train, 'lstnet')
    
    # Train TCN model
    reset_random_seeds()
    tcn_model = build_tcn(input_shape)
    train_model(tcn_model, X_train, y_train, 'tcn')

if __name__ == "__main__":
    main()
