import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import matplotlib.pyplot as plt

spy_data = pd.read_csv("spy_max.csv")
spy_data = spy_data.iloc[::-1].reset_index(drop=True)
spy_data = spy_data.drop(columns=['Date'])

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(spy_data.drop(columns=['Close/Last']))

label_scaler = MinMaxScaler()
scaled_labels = label_scaler.fit_transform(spy_data[['Close/Last']]).flatten()

def create_sequence(data, labels, seq_length, output_length):
    X = []
    y = []
    for i in range(len(data) - seq_length - output_length + 1):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length:i + seq_length + output_length])
    return np.array(X), np.array(y)

seq_len = 10
output_len = 1  

X, y = create_sequence(scaled_features, scaled_labels, seq_len, output_len)

X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                   return_sequences=False,
                   input_shape=(seq_len, X.shape[2])))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    model.add(Dense(units=output_len))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    directory='keras_tuner_dir',
    project_name='lstm_hyperparameter_tuning'
)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rmse_values = []

for fold_index, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Training fold {fold_index + 1}/{n_splits}")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])

    best_model = tuner.get_best_models(num_models=1)[0]

    val_loss = best_model.evaluate(X_val, y_val)
    print(f"Fold {fold_index + 1}/{n_splits} - Validation Loss: {val_loss}")

    if fold_index == n_splits - 1:
        X_train_final, X_test, y_train_final, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        # Train the best model on the full training set
        history = best_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

        predictions = best_model.predict(X_test)

        predictions = label_scaler.inverse_transform(predictions)
        y_test_actual = label_scaler.inverse_transform(y_test)

        rmse = np.sqrt(np.mean((predictions - y_test_actual) ** 2))
        rmse_values.append(rmse)


        ##############################


        plt.figure(figsize=(10, 6))
        plt.plot(y_test_actual, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.title('Predicted vs Actual Prices')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_actual, predictions, alpha=0.5)
        plt.title('Predicted vs Actual Prices')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.grid(True)
        plt.show()

        residuals = predictions - y_test_actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_actual, residuals, alpha=0.5)
        plt.title('Residuals vs Actual Prices')
        plt.xlabel('Actual Prices')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor='k')
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()






        ##############################

print(f'Average RMSE: {np.mean(rmse_values)}')
