import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers
from matplotlib import pyplot as plt


def binary_classification():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    train_df = pd.read_csv('./data/california_housing_train.csv')
    test_df = pd.read_csv('./data/california_housing_test.csv')
    train_df = train_df.reindex(np.random.permutation(train_df.index))

    # normalize
    train_df_mean = train_df.mean()
    train_df_std = train_df.std()
    train_df_norm = (train_df - train_df_mean) / train_df_std

    test_df_mean = test_df.mean()
    test_df_std = test_df.std()
    test_df_norm = (test_df - test_df_mean) / test_df_std

    # add binary column
    threshold = 265000
    train_df_norm['median_house_value_is_high'] = (train_df['median_house_value'] > threshold).astype(float)
    test_df_norm['median_house_value_is_high'] = (test_df['median_house_value'] > threshold).astype(float)

    # add feature layer
    feature_columns = []
    median_income = tf.feature_column.numeric_column('median_income')
    feature_columns.append(median_income)
    total_rooms = tf.feature_column.numeric_column('total_rooms')
    feature_columns.append(total_rooms)
    feature_layer = layers.DenseFeatures(feature_columns)

    # train the model
    learning_rate = 0.001
    epochs = 20
    batch_size = 100
    label_name = 'median_house_value_is_high'
    classification_threshold = 0.5
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold),
        tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'),
        tf.keras.metrics.Recall(thresholds=classification_threshold, name='recall'),
        tfa.metrics.F1Score(num_classes=1, threshold=classification_threshold, name='f1_score'),
        tf.keras.metrics.AUC(num_thresholds=100, name='auc')
    ]
    model = create_model(learning_rate, feature_layer, METRICS)
    epochs, hist = train_model(model, train_df_norm, epochs, label_name, batch_size)
    plot_curve(epochs, hist, ['accuracy', 'precision', 'recall', 'f1_score', 'auc'])

    # evaluate the model
    features = {name: np.array(value) for name, value in test_df_norm.items()}
    label = np.array(features.pop(label_name))
    model.evaluate(x=features, y=label, batch_size=batch_size)


def create_model(my_learning_rate, feature_layer, my_metrics):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=my_metrics)
    return model


def train_model(model, dataset, epochs, label_name,
                batch_size=None, shuffle=True):
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=shuffle)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


print('Defined the create_model and train_model functions.')


def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()


print('Defined the plot_curve function.')
