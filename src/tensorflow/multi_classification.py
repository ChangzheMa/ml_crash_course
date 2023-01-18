import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt


def multi_classification():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format
    np.set_printoptions(linewidth=200)

    # load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # view the data
    # plt.imshow(X=x_train[2917], cmap='gray', vmin=0, vmax=255)
    # plt.show()
    print(x_train[2917])
    print(x_train[2917][10])
    print(x_train[2900][10][16])

    # normalize the data
    x_train_norm = x_train / 255.0
    x_test_norm = x_test / 255.0
    print(x_train_norm[2900][10])

    # make the multi classification model
    learning_rate = 0.003
    epochs = 500
    batch_size = 4000
    validation_split = 0.2
    model = create_model(learning_rate)
    epochs, hist = train_model(model, train_feature=x_train_norm, train_label=y_train, epochs=epochs,
                               batch_size=batch_size, validation_split=validation_split)
    list_of_metrics_to_plot = ['accuracy']
    plot_curve(epochs, hist, list_of_metrics_to_plot)
    print('\n Evaluate the new model against the test set:')
    model.evaluate(x=x_test_norm, y=y_test, batch_size=batch_size)


def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    plt.legend()
    plt.show()


def create_model(learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=900, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.7))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy')
    return model


def train_model(model, train_feature, train_label, epochs, batch_size, validation_split=0.1):
    history = model.fit(x=train_feature, y=train_label, batch_size=batch_size, epochs=epochs, shuffle=True,
                        validation_split=validation_split)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist
