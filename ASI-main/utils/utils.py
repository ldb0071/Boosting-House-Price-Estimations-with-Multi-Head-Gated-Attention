
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Lambda

# calculates the square root of the square of a tensor maintaining the dimension (m,s1,n) --> (m,s1,1)
def srs(tensor):

    tensor_srs = Lambda(lambda x: tf.math.sqrt(
        tf.math.reduce_sum(
            tf.math.square(x), axis=-1, keepdims=True
        )
    ))(tensor)

    return tensor_srs


def load_model(X: None, y: None, seed: None, test_size: None, model: None):
    # split data into train and test sets
    seed = seed
    test_size = test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # load model from file
    loaded_model = joblib.load(model)

    # predictions
    predictions = loaded_model.predict(X_test)

    # evaluate predictions
    return r2_score(y_test, predictions), mean_squared_error(y_test, predictions), mean_absolute_error(y_test,
                                                                                                       predictions)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

class BatchPreProcessor(object):

    def normalize(self, x, vmax, vmin):
        self.v = (x - vmin) / (vmax - vmin)
        return self.v

    def normalize_a_b(self, x, vmax, vmin, a, b):
        self.v = (b - a) * ((x - vmin) / (vmax - vmin)) + a
        return self.v

    def positive(self, x):
        return -x if x < 0 else x

    def normalize_matrix(self, x):
        self.f = np.vectorize(self.normalize, otypes=[np.float])
        for i in range(x.shape[1]):
            self.vmax = np.max(x[:, i])
            self.vmin = np.min(x[:, i])
            x[:, i] = self.f(x[:, i], self.vmax, self.vmin)
        return x

    def normalize_matrix_a_b(self, x, a, b):
        self.f = np.vectorize(self.normalize_a_b, otypes=[np.float])
        self.vmax = np.max(x)
        self.vmin = np.min(x)
        for c in range(x.shape[1]):
            for r in range(x.shape[0]):
                x[r, c] = self.normalize_a_b(x[r, c], self.vmax, self.vmin, a, b)
        return x

    def positive_matrix(self, x):
        self.f = np.vectorize(self.positive, otypes=[np.float])
        for c in range(x.shape[1]):
            for r in range(x.shape[0]):
                x[r, c] = self.f(x[r, c])
        return x
