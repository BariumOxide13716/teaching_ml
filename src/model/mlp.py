import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

class MLP():
    def __init__(self,
                 hidden_layers: list[int] = [ 10 ],
                 activation: list[str] = [ 'relu' ],
                 learning_rate: float = 0.001,
                 epochs: int = 10,
                 test_size: float = 0.2,
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
        self.random_seed = random_seed

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=self.test_size, 
                                                            random_state=self.random_seed)

        # Build the model if it hasn't been built yet
        # otherwise use the model in the class

        if self.model is None:
            model = Sequential() if self.model is None else self.model

            for i in range(len(self.hidden_layers)):
                if i == 0:
                    model.add(Dense(self.hidden_layers[i], 
                                    activation=self.activation[i], 
                                    input_shape=(X.shape[1],)))
                else:
                    model.add(Dense(self.hidden_layers[i], 
                                    activation=self.activation[i]))

            model.add(Dense(1, 
                            activation=self.activation[-1]))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        else:
            model = self.model

        
        #model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test))

        self.model = model