import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import os
import numpy as np

class MLP():
    def __init__(self,
                 hidden_layers: list[int] = [ 10 ],
                 activation: list[str] = [ 'relu' ],
                 learning_rate: float = 0.001,
                 epochs: int = 10,
                 test_size: float = 0.2,
                 batch_size: int = 32,
                 random_seed: int = 42):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.n_layer = len(hidden_layers)
        assert len(activation) == self.n_layer + 1, \
            "The number of activation functions should be equal to the number of layers + 1 (for the output layer)."

        self.model = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed

    def load_model(self,
                   path: str):
        assert os.path.exists(path), \
            f"Model file not found at {path}. Please provide a valid path to load the model."
        self.model = tf.keras.models.load_model(path)
    
    def save_model(self,
                   path: str):
        assert self.model is not None, \
            "Model has not been trained yet. Please train the model before saving."
        self.model.save(path)

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray):
        X = X.reshape(-1, 1) if len(X.shape) == 1 else X

        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=self.test_size, 
                                                            random_state=self.random_seed)
    
        print(f"========Training data size: {len(X_train)}, Testing data size: {len(X_test)}")
        print(f"========Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
        if self.model is not None:
            self.model.summary()
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                               loss='mean_squared_error')
            print("Model already exists. Training will continue from the existing model.")
        else:
            self.model = Sequential()
            for i in range(self.n_layer):
                if i == 0:
                    print(f"========Adding layer {i+1} with {self.hidden_layers[i]} neurons and '{self.activation[i]}' activation function (input layer).")
                    self.model.add(Dense(self.hidden_layers[i], 
                                         activation=self.activation[i], 
                                         input_shape=(X_train.shape[1],)
                                         )
                                   )
                else:
                    print(f"========Adding layer {i+1} with {self.hidden_layers[i]} neurons and '{self.activation[i]}' activation function.")
                    self.model.add(Dense(self.hidden_layers[i], 
                                         activation=self.activation[i]))
            self.model.add(Dense(1, 
                                 activation=self.activation[-1]))
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                               loss='mean_squared_error')
        print("Starting training...")
        if len(X_train) < 50:
            print(f"========training data details:")
            print(X_train)
            print(y_train)
        self.model.fit(X_train, 
                       y_train, 
                       epochs=self.epochs, 
                       validation_data=(X_test, y_test),
                       batch_size=self.batch_size)
    
    def predict(self, X):
        assert self.model is not None, \
            "Model has not been trained yet. Please train the model before making predictions."
        X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        return self.model.predict(X)
    
def quick_mlp(X, 
                y, 
                model_path=None,
                hidden_layers=[20, 20], 
                activation=['relu', 'sigmoid', 'linear'], 
                learning_rate=0.001, 
                test_size=0.2,
                random_seed=42,
                epochs=100):
    mlp = MLP(hidden_layers=hidden_layers,
              activation=activation,
              learning_rate=learning_rate,
              test_size=test_size,
              random_seed=random_seed,
              epochs=epochs)
    if model_path is not None:
        if os.path.exists(model_path):
            mlp.load_model(model_path)
        else:
            print(f"Model file not found at {model_path}. A new model will be trained.")
    mlp.fit(X, y)
    if model_path is not None:
        mlp.save_model(model_path)
    return mlp
        

