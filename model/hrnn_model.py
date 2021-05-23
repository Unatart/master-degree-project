import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import activations

from get_features.tokenizer.tokenizer import create_numerate_array
from get_features.image_character.image_feature import show

tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel('ERROR')


id_size = 1
features_size = 7
vocab_size = 458
max_scenes_size = 396
embedding_dim = 458
rnn_units = 512


class ScenesModel(tf.keras.Model):
    def __init__(self, rnn_units):
        super().__init__(self)
        self.normalization = tf.keras.layers.BatchNormalization()
        self.gru = tf.keras.layers.GRU(rnn_units, return_state=True)
        self.dense = tf.keras.layers.Dense(rnn_units)

    def call(self, _input, states=None, return_state=True, training=False):
        norm = self.normalization(_input)
        if states is None:
            states = self.gru.get_initial_state(norm)
        x, final_state = self.gru(norm, initial_state=states, training=training)
        result = self.dense(final_state)

        return result


class HRNNModel(tf.keras.Model):
    def __init__(self, max_scenes_size, vocab_size, rnn_units, scenes_model):
        super().__init__(self)
        self.scenes_model = scenes_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, max_scenes_size)
        self.time_distr = tf.keras.layers.TimeDistributed(self.scenes_model)
        self.gru_sess = tf.keras.layers.GRU(rnn_units, return_sequences=True)
        self.concat = tf.keras.layers.Concatenate()
        self.dense = tf.keras.layers.Dense(vocab_size, activation=activations.softmax)

    def call(self, _input, states=None, return_state=False, training=False):
        features = self.time_distr(_input[1])
        ids = self.embedding(_input[0])
        x = self.concat([ids, features, _input[2]])
        id_x = self.gru_sess(x, training=training)

        result = self.dense(id_x)

        return result

    def predict(self, x, history, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False, ):
        predictions = super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers,
                                      use_multiprocessing)
        predicted_values = []
        for i in range(len(predictions)):
            output = predictions[-1][0]
            mask = []
            for i in range(vocab_size):
                if i + 1 in history:
                    mask.append(1)
                else:
                    mask.append(0)
            masked_prediction = np.ma.array(output, mask=mask)
            predicted_values.append(np.argmax(masked_prediction) + 1)

        return predicted_values


def toNumber(_str, bias):
    return len(_str) + bias


def getVideoInfo(info_df, mean=False):
    scenes_array = [
        info_df['brightness'].to_numpy(),
        info_df['colorfulness'].to_numpy(),
        info_df['energy'].to_numpy(),
        info_df['tempo'].to_numpy(),
        info_df['amplitude'].to_numpy(),
        info_df['mfcc'].to_numpy(),
        create_numerate_array(info_df['hue'].to_numpy())
    ]
    scenes_array = np.nan_to_num(scenes_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler((-1, 1))
    scenes_array = scaler.fit_transform(scenes_array)

    if mean:
        new_scenes_array = []
        for i in range(len(scenes_array)):
            new_scenes_array.append(np.around(np.mean(scenes_array[i]), 3))
        scenes_array = new_scenes_array

    return scenes_array


def getMovieInfo(trailers_row):
    key = trailers_row['Place of act'].keys()[0]
    emotions = json.loads(trailers_row['emotions'][key].replace("\'", "\""))

    return [
        np.nan_to_num([trailers_row['imdb_rating'][key]], copy=False, nan=0.0, posinf=0.0, neginf=0.0)[0],
        np.around(emotions["negative"], 3),
        np.around(emotions["neutral"], 3),
        np.around(emotions["positive"], 3),
    ]


def getScenesArray(movies_id, trailers_df):
    info = {}
    scenes_array = []
    movieinfo_array = []
    info_folder = "/Volumes/Seagate/natasha-diploma/videoinfo"
    for i in range(0, len(movies_id)):
        info_csv = info_folder + '/' + movies_id[i] + '.csv'
        info_df = pd.read_csv(info_csv, index_col=None, header=0)
        info[movies_id[i]] = getVideoInfo(info_df.iloc[:, 4:])
        scenes_array.append(info[movies_id[i]])
        movieinfo_array.append(getMovieInfo(trailers_df.loc[trailers_df['trailers_name'] == movies_id[i]]))

    return scenes_array, movieinfo_array


def pad_nested_arrays(seq, maxlen_sent):
    new_seq = []
    for i in range(len(seq)):
        size = len(seq[i])
        if maxlen_sent == size:
            new_seq.append(np.pad(seq[i], 0, 'mean'))
        else:
            size_beg = (maxlen_sent - size) // 2
            size_end = size_beg + 1
            new_seq.append(np.pad(seq[i], (size_beg, size_end), 'mean')[:396])

    return new_seq


def setup_model():
    trailers_csv = "/Users/nutochkina/Desktop/VKR/trailers.csv"
    trailers_df = pd.read_csv(trailers_csv, index_col=None, header=0)
    trailers_df.drop(trailers_df.columns[trailers_df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)

    sessions_csv = "/Users/nutochkina/Desktop/VKR/sessions1.csv"
    sessions_df = pd.read_csv(sessions_csv, index_col=None, header=0)
    sessions_df.drop(sessions_df.columns[sessions_df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)

    movies_id = sessions_df.to_numpy()
    copy = movies_id
    np.random.shuffle(copy)
    movies_id = np.concatenate((np.array(movies_id).flatten(), copy.flatten()))

    vocab = np.unique(movies_id)
    vocab_len = len(vocab)

    tokenized = create_numerate_array(movies_id)
    scenes_array, movieinfo_array = getScenesArray(movies_id, trailers_df)
    padded = np.stack([pad_nested_arrays(r, max_scenes_size) for r in scenes_array], axis=0)

    train_tokenized = tokenized[:4500]
    test_tokenized = tokenized[4500:7460]

    train_scenes = padded[:4500]
    test_scenes = padded[4500:7460]

    train_movinfo = movieinfo_array[:4500]
    test_movinfo = movieinfo_array[4500:7460]

    ids_ds = tf.data.Dataset.from_tensor_slices(train_tokenized)
    features_ds = tf.data.Dataset.from_tensor_slices(train_scenes)
    movie_ds = tf.data.Dataset.from_tensor_slices(train_movinfo)

    ids_ds_test = tf.data.Dataset.from_tensor_slices(test_tokenized)
    features_ds_test = tf.data.Dataset.from_tensor_slices(test_scenes)
    movie_ds_test = tf.data.Dataset.from_tensor_slices(test_movinfo)

    ds_test = tf.data.Dataset.zip((ids_ds_test, features_ds_test, movie_ds_test))
    ds = tf.data.Dataset.zip((ids_ds, features_ds, movie_ds))

    seq_length = 10

    sequences = ds.batch(seq_length + 1, drop_remainder=True)
    sequences_test = ds_test.batch(seq_length + 1, drop_remainder=True)

    def full_split_input_target(id_sequence, feature_sequence, movie_sequence):
        input_text = (id_sequence[:-1], feature_sequence[:-1], movie_sequence[:-1])
        target_text = (id_sequence[1:])
        return input_text, target_text

    ds = sequences.map(full_split_input_target)
    ds_test = sequences_test.map(full_split_input_target)

    BATCH_SIZE = 8
    TEST_BATCH_SIZE = 1
    BUFFER_SIZE = 10000

    ds = (
        ds
            .shuffle(BUFFER_SIZE)
            .batch(TEST_BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

    ds_test = (
        ds_test
            .shuffle(BUFFER_SIZE)
            .batch(TEST_BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

    scenes_model = ScenesModel(rnn_units=16)
    scenes_model.build(input_shape=(seq_length, features_size, max_scenes_size))
    scenes_model.summary()

    model = HRNNModel(
        max_scenes_size=max_scenes_size,
        vocab_size=vocab_size,
        rnn_units=rnn_units,
        scenes_model=scenes_model
    )

    for _input, _output in ds.take(1):
        model(_input)

    model.summary()

    return model, ds, ds_test, vocab


def train_model():
    log_file = "/Users/nutochkina/Desktop/VKR/hrnn_model/history.ckpt"
    model, ds, ds_test, _ = setup_model()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_file, save_weights_only=True, verbose=1)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
        name='sparse_categorical_accuracy', dtype=None
    )])

    EPOCHS = 15
    model.fit(ds, validation_data=ds_test, epochs=EPOCHS, callbacks=[cp_callback])


def evaluate_and_predict():
    log_dir = "/Users/nutochkina/Desktop/VKR/hrnn_model/"
    model, ds, ds_test, vocab = setup_model()
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None)])

    latest = tf.train.latest_checkpoint(log_dir)
    model.load_weights(latest)
    model.evaluate(ds_test, batch_size=1, return_dict=True)

    for _input, _output in ds_test.take(1):
        values = model.predict(_input, _input[0].numpy())
        show(vocab[_input[0].numpy()[0][-1] - 1])
        show(vocab[values[0] - 1])


