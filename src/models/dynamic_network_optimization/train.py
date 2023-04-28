import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_shape, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train):
    model = build_model(X_train.shape[1:], y_train.shape[1])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return model, history

