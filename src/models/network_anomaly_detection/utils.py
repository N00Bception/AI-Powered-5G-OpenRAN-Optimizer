import numpy as np
from sklearn.preprocessing import StandardScaler

class DataUtils:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, x):
        x = self.scaler.fit_transform(x)
        return x

    def split_data(self, x, y, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        x_train = self.preprocess_data(x_train)
        x_test = self.preprocess_data(x_test)
        y_train = np.asarray(y_train).astype('float32')
        y_test = np.asarray(y_test).astype('float32')
        return x_train, x_test, y_train, y_test
