import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns


def neural_nets():
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format
    print("Imported modules.")

    # load data
    train_df = pd.read_csv('./data/california_housing_train.csv')
    test_df = pd.read_csv('./data/california_housing_test.csv')
    train_df = train_df.reindex(np.random.permutation(train_df.index))

    # normalization data
    train_df_mean = train_df.mean()
    train_df_std = train_df.std()
    train_df_norm = (train_df - train_df_mean) / train_df_std

    test_df_mean = test_df.mean()
    test_df_std = test_df.std()
    test_df_norm = (test_df - test_df_mean) / test_df_std

    # create feature_layer
    resolution_in_zs = 0.3
    feature_layer = generate_feature_layer(train_df_norm, resolution_in_zs)

    # make a linear model
    # make_linear_model(feature_layer, train_df_norm, test_df_norm)

    # make a neural model
    make_neural_model(feature_layer, train_df_norm, test_df_norm)


def make_neural_model(feature_layer, train_df_norm, test_df_norm):
    learning_rate = 0.01
    epochs = 1000
    batch_size = 1000
    label_name = 'median_house_value'
    model = create_neural_model(learning_rate, feature_layer)
    epochs, mse = train_model(model, train_df_norm, epochs, batch_size, label_name)
    plot_the_loss_curve(epochs, mse)
    # test the neural model
    test_features = {name: np.array(value) for name, value in test_df_norm.items()}
    test_label = np.array(test_features.pop(label_name))
    print('\n Evaluate the neural model against the test set:')
    model.evaluate(x=test_features, y=test_label, batch_size=batch_size)


def make_linear_model(feature_layer, train_df_norm, test_df_norm):
    learning_rate = 0.01
    epochs = 15
    batch_size = 1000
    label_name = 'median_house_value'
    model = create_model(learning_rate, feature_layer)
    epochs, mse = train_model(model, train_df_norm, epochs, batch_size, label_name)
    plot_the_loss_curve(epochs, mse)
    # test the linear model
    test_features = {name: np.array(value) for name, value in test_df_norm.items()}
    test_label = np.array(test_features.pop(label_name))
    print('\n Evaluate the linear regression model against the test set:')
    model.evaluate(x=test_features, y=test_label, batch_size=batch_size)


def generate_feature_layer(train_df_norm, resolution_in_zs=0.3):
    feature_columns = []

    latitude_as_numeric = tf.feature_column.numeric_column('latitude')
    latitude_bound = list(np.arange(
        int(min(train_df_norm['latitude'])), int(max(train_df_norm['latitude'])), resolution_in_zs
    ))
    latitude = tf.feature_column.bucketized_column(latitude_as_numeric, latitude_bound)

    longitude_as_numeric = tf.feature_column.numeric_column('longitude')
    longitude_bound = list(np.arange(
        int(min(train_df_norm['longitude'])), int(max(train_df_norm['longitude'])), resolution_in_zs
    ))
    longitude = tf.feature_column.bucketized_column(longitude_as_numeric, longitude_bound)

    lat_x_longi = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
    crossed_feature = tf.feature_column.indicator_column(lat_x_longi)
    feature_columns.append(crossed_feature)

    median_income = tf.feature_column.numeric_column('median_income')
    feature_columns.append(median_income)

    population = tf.feature_column.numeric_column('population')
    feature_columns.append(population)

    return tf.keras.layers.DenseFeatures(feature_columns)


def plot_the_loss_curve(epochs, mse):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')

    plt.plot(epochs, mse, label='Loss')
    plt.legend()
    plt.ylim([mse.min()*0.95, mse.max()*1.03])
    plt.show()


def create_model(my_learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=True)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist['mean_squared_error']

    return epochs, rmse


def create_neural_model(learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=5, activation='relu', name='Hidden1',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(units=5, activation='relu', name='Hidden2',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(units=1, name='Output'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
    return model
