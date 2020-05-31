from LoadData import load_data
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
tf.enable_eager_execution()

# This script uses a deep neural network, build with Tensorflow2, to predict the car prices


class NeuralNetworkModel(Model):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__()
        self.dense1 = Dense(16, activation='relu')
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(32, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(16, activation='relu')
        self.dropout3 = Dropout(0.1)
        self.dense4 = Dense(1, activation='sigmoid')

    def call(self, x, training=True):
        if training:
            x = self.dropout1(self.dense1(x))
            x = self.dropout2(self.dense2(x))
            x = self.dropout3(self.dense3(x))
            x = self.dense4(x)
        else:
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.dense4(x)
        return x


data = load_data().to_numpy()

X_train = data[:100, :-1]
Y_train = data[:100, -1].reshape((-1, 1))
X_test = data[100:, :-1]
Y_test = data[100:, -1]

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)).shuffle(10000).batch(16)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(16)


nn_model = NeuralNetworkModel()

loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = nn_model(images, training=True)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, nn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, nn_model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(input_data, true_output):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = nn_model(input_data, training=False)
    t_loss = loss_function(true_output, predictions)

    test_loss(t_loss)
    test_accuracy(true_output, predictions)


EPOCHS = 100

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for x, y in train_ds:
        train_step(x, y)

    for x, y in test_ds:
        test_step(x, y)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

