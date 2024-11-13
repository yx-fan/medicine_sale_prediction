# model_selection.py
import numpy as np
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def train_sarimax_model(history_y, history_exog):
    return pm.auto_arima(
        history_y, exogenous=history_exog, seasonal=True, m=52,
        max_d=2, max_p=3, max_q=3, D=1, stepwise=True, trace=True
    )

def calculate_residuals(log_y, exog, auto_model, train_end):
    residuals = []
    for t in range(train_end):
        exog_current = exog.iloc[t: t+1].fillna(0)
        next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
        actual_y_at_t = log_y.iloc[t]
        auto_model.update([actual_y_at_t], exogenous=exog_current)
        residual = actual_y_at_t - next_pred_log
        residuals.append(residual)
    return residuals

def fit_random_forest(history_exog, residuals):
    rf_model = RandomForestRegressor()
    rf_model.fit(history_exog, residuals)
    return rf_model

def create_cnn_model(input_shape):
    # Define CNN model with explicit input layer
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=16, kernel_size=1, activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def fit_cnn_model(history_exog, history_y, residuals, epochs=20, batch_size=4):
    history_exog_with_y = np.concatenate([history_exog, history_y.values.reshape(-1, 1)], axis=1)
    history_exog_reshaped = np.expand_dims(history_exog_with_y, axis=-1)

    cnn_model = create_cnn_model(input_shape=(history_exog_reshaped.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    cnn_model.fit(
        history_exog_reshaped, np.array(residuals),
        epochs=epochs, batch_size=batch_size, validation_split=0.2,
        callbacks=[early_stopping], verbose=1
    )
    return cnn_model
