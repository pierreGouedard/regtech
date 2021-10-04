from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping


class Commit2Test():
    """Match distributed representation of commits to distributed representation of tests"""

    def __init__(
            self, latent_test_dim: int, n_kernels: int, kernel_size: int, input_dim: Tuple[int, int],
            learning_rate: float, nb_epoch: int, batch_size: int = 32
    ):
        # training params
        self.learning_rate, self.nb_epoch, self.batch_size = learning_rate, nb_epoch, batch_size
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

        # I/O dim
        self.latent_test_dim, self.input_dim = latent_test_dim, input_dim

        # Kernel's dim
        self.n_kernels, self.kernel_size = n_kernels, kernel_size

        # Create and compile model
        self.network = self.__create_network(self.latent_test_dim, self.n_kernels, self.kernel_size, self.input_dim)
        self.network.compile(optimizer=self.optimizer, loss=MeanSquaredError())

        # Evaluation attribute
        self.history = None

    @staticmethod
    def __create_network(
            latent_dim: int, n_kernels: int, kernel_size: int, input_dim: Tuple[int, int]
    ) -> Model:
        """
        Implement the forward propagation for the encoding layers:
        CONV1D -> RELU -> MAXPOOL -> DENSE -> OUTPUT

        inputs dim are (n_batch, n_steps, n_embedding)

        In the case of chain of character encoding n_embedding = 36 and n_steps is large enough so that most
        of the sentence won't be truncated

        """
        X_input = layers.Input(input_dim)

        # 1D conv with activation
        X = layers.Conv1D(
            n_kernels, kernel_size, activation='relu', padding="same", input_shape=input_dim
        )(X_input)

        # Max pool layer
        X = layers.MaxPooling1D(pool_size=input_dim[0], padding='valid')(X)

        # Flatten
        X = layers.Flatten()(X)

        # End with a dense FC layer
        X = layers.Dense(latent_dim, name='output_layer')(X)

        network = Model(inputs=X_input, outputs=X, name='model_embedding')

        return network

    def fit(self, X_train, y_train, show_eval: bool = True) -> None:
        """
        Fit network using early stopping on validation dataset.

        Args:
            X_train: np.ndarray
            y_train: np.ndarray
            show_eval: bool

        Returns:
        None
        """
        # Build datasets
        train_dataset, val_dataset = self.build_datasets(X_train, y_train)

        # Create early stopping callback
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

        # fit the model
        self.history = self.network.fit(
            train_dataset, validation_data=val_dataset, epochs=self.nb_epoch, batch_size=self.batch_size,
            callbacks=[es]
        )

        if show_eval:
            self.evaluate(train_dataset, val_dataset)

    def evaluate(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> None:
        """
        Display performance of model on train/val and history of metrics through epochs.

        Args:
            train_dataset: tf.data.Dataset
                Training dataset.
            val_dataset: tf.data.Dataset
                Validation dataset

        Returns:
        None
        """

        # Get metrics
        train_mse = self.network.evaluate(train_dataset, verbose=0)
        test_mse = self.network.evaluate(val_dataset, verbose=0)

        print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

        # plot training history
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

    def build_datasets(self, X: np.ndarray, y: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Build train and validation dataset.

        Parameters
        ----------
        X: ndarray
            Features array.
        y: ndarray
            Target array.

        Returns
        -------
        tensorflow dataset
            Train and val dataset composed of input and labels.
        """
        # Build final TF dataset
        all_dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(X), tf.data.Dataset.from_tensor_slices(y),
        ))\
            .shuffle(buffer_size=32)

        train_dataset = all_dataset.skip(int(len(all_dataset) * 0.05))\
            .batch(self.batch_size, drop_remainder=False) \
            .prefetch(int(self.batch_size / 4))

        val_dataset = all_dataset.take(int(len(all_dataset) * 0.05))\
            .batch(self.batch_size, drop_remainder=False) \
            .prefetch(int(self.batch_size / 4))

        return train_dataset, val_dataset


if __name__ == "__main__":
    # Network parameters
    input_dim, latent_test_dim = (20, 100), 25
    n_kernels, kernel_size = 100, 5

    # Learning parameters
    learning_rate, nb_epoch = 0.01, 1000

    # Instantiat model
    c2t_model = Commit2Test(latent_test_dim, n_kernels, kernel_size, input_dim, learning_rate, nb_epoch)

    # Example of dataset => building 5 random matrices of shape (1000, 36) [final shape (5, 1000, 36)] :
    X = np.stack([np.vstack([np.random.randn(input_dim[1]) for i in range(input_dim[0])]) for j in range(1000)])
    y = np.stack([np.random.randn(latent_test_dim) for j in range(1000)])

    # Fit model
    c2t_model.fit(X, y)


