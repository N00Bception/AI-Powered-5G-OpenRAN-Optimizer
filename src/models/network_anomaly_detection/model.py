import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class NetworkAnomalyDetectionModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._create_model()

    def _create_model(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=self.input_shape),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

    def predict(self, x):
        return self.model.predict(x)
