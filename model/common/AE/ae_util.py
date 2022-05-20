import tensorflow.keras as keras



def make_model_LSTM(seq_size, features_size, input_shape):
    size_1_layer_size = 18
    size_2_layer_size = 12
    middle_layer_size = 10
    # define model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(size_1_layer_size, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.LSTM(size_2_layer_size, return_sequences=False))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.RepeatVector(seq_size))
    model.add(keras.layers.LSTM(middle_layer_size, return_sequences=True, name='embedding'))
    model.add(keras.layers.LeakyReLU(alpha=0.1, name='leaky_embedding'))
    model.add(keras.layers.LSTM(size_2_layer_size, return_sequences=True))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.LSTM(size_1_layer_size, return_sequences=True))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(features_size)))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'], weighted_metrics=['mse'])
    model.summary()

    return model


def make_model_LSTM_SLIM(seq_size, features_size):
    inputs = keras.layers.Input(shape=(seq_size, features_size))
    encoded = keras.layers.LSTM(int(features_size/2), name='embedding')(inputs)

    decoded = keras.layers.RepeatVector(seq_size)(encoded)
    decoded = keras.layers.LSTM(features_size, return_sequences=True)(decoded)

    sequence_autoencoder = keras.models.Model(inputs, decoded)
    encoder = keras.models.Model(inputs, encoded)
    sequence_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    sequence_autoencoder.summary()
    return sequence_autoencoder

def make_model_CNN_AE(seq_size, features_size):
    input_window = keras.layers.Input(shape=(seq_size, features_size))
    x = keras.layers.Conv1D(16, 3, activation="relu", padding="same")(input_window)  # 10 dims
    # x = BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)  # 5 dims
    x = keras.layers.Conv1D(1, 3, activation="relu", padding="same")(x)  # 5 dims
    # x = BatchNormalization()(x)
    encoded = keras.layers.MaxPooling1D(2, padding="same", name='embedding')(x)  # 3 dims

    encoder = keras.models.Model(input_window, encoded)

    # 3 dimensions in the encoded layer

    x = keras.layers.Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 3 dims
    # x = BatchNormalization()(x)
    x = keras.layers.UpSampling1D(2)(x)  # 6 dims
    x = keras.layers.Conv1D(16, 1, activation='relu')(x)  # 5 dims
    # x = BatchNormalization()(x)
    x = keras.layers.UpSampling1D(2)(x)  # 10 dims
    decoded = keras.layers.Conv1D(features_size, 3, activation='sigmoid', padding='same')(x)  # 10 dims
    autoencoder = keras.models.Model(input_window, decoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()
    return autoencoder
# def

models_dict = {'helpers': make_model_LSTM, 'LSTM_SLIM': make_model_LSTM_SLIM, 'CNN': make_model_CNN_AE}