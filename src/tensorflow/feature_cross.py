import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from matplotlib import pyplot as plt


def generate_fp_layer():
    feature_columns = []
    latitude = tf.feature_column.numeric_column('latitude')
    feature_columns.append(latitude)
    longitude = tf.feature_column.numeric_column('longitude')
    feature_columns.append(longitude)
    fp_feature_layer = layers.DenseFeatures(feature_columns)
    return fp_feature_layer


def generate_buckets_layer(train_df, resolution_in_degrees=1.0):
    feature_columns = []

    latitude_as_numeric_column = tf.feature_column.numeric_column('latitude')
    latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
                                         int(max(train_df['latitude'])), resolution_in_degrees))
    latitude = tf.feature_column.bucketized_column(latitude_as_numeric_column, latitude_boundaries)

    longitude_as_numeric_column = tf.feature_column.numeric_column('longitude')
    longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
                                          int(max(train_df['longitude'])), resolution_in_degrees))
    longitude = tf.feature_column.bucketized_column(longitude_as_numeric_column, longitude_boundaries)

    feature_columns.append(latitude)
    feature_columns.append(longitude)

    return layers.DenseFeatures(feature_columns)


def generate_cross_layer(train_df, resolution_in_degrees=1.0):
    feature_columns = []

    latitude_as_numeric_column = tf.feature_column.numeric_column('latitude')
    latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
                                         int(max(train_df['latitude'])), resolution_in_degrees))
    latitude = tf.feature_column.bucketized_column(latitude_as_numeric_column, latitude_boundaries)

    longitude_as_numeric_column = tf.feature_column.numeric_column('longitude')
    longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
                                          int(max(train_df['longitude'])), resolution_in_degrees))
    longitude = tf.feature_column.bucketized_column(longitude_as_numeric_column, longitude_boundaries)

    lat_x_lgi = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
    crossed_feature = tf.feature_column.indicator_column(lat_x_lgi)
    feature_columns.append(crossed_feature)

    return layers.DenseFeatures(feature_columns)


def feature_cross():
    train_df = pd.read_csv('./data/california_housing_train.csv')
    test_df = pd.read_csv('./data/california_housing_test.csv')

    scale_factor = 1000.0
    train_df['median_house_value'] /= scale_factor
    test_df['median_house_value'] /= scale_factor

    train_df = train_df.reindex(np.random.permutation(train_df.index))

    # 第一个过滤方法
    # fp_feature_layer = generate_fp_layer()

    # 第二个过滤方法
    resolution_in_degrees = 0.4
    buckets_feature_layer = generate_buckets_layer(train_df, resolution_in_degrees)
    cross_feature_layer = generate_cross_layer(train_df, resolution_in_degrees)

    learning_rate = 0.05
    epochs = 50
    batch_size = 100
    label_name = 'median_house_value'

    model = create_model(learning_rate, cross_feature_layer)
    epochs, rmse = train_model(model, train_df, epochs, batch_size, label_name)
    plot_the_loss_curve(epochs, rmse)

    print('\n: Evaluate the new model against the test set:')
    test_features = {name: np.array(value) for name, value in test_df.items()}
    test_label = np.array(test_features.pop(label_name))
    model.evaluate(x=test_features, y=test_label, batch_size=batch_size)


def create_model(my_learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    # feature is :
    # {
    #   'key1': [1.0, 1.2, 4.2, -1.1, ...],
    #   'key2': [-0.2, 1.1, 9.0, ...]
    # }
    # label is ：[-0.2, 2.1, 1.1, 2.6, ...]
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)
    epochs = history.epoch

    hist = pd.DataFrame(history.history)
    rmse = hist['root_mean_squared_error']

    return epochs, rmse


def plot_the_loss_curve(epochs, rmse):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Squared Error')

    plt.plot(epochs, rmse, label='Loss')
    plt.legend()
    plt.ylim([rmse.min() * 0.94, rmse.max() * 1.05])
    plt.show()


print('Defined the create_model, train_model, and plot_the_loss_curve functions.')
