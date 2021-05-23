import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import activations
import json
from get_features.tokenizer.tokenizer import create_numerate_array

tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel('ERROR')

id_size = 1
id_embed_size = 11
vocab_size = 458
rnn_units = 512


def toNumber(_str, bias):
    return len(_str) + bias


def getVideoInfo(info_df, trailers_row, mean=True):
    tokenized_hue = create_numerate_array(info_df['hue'].to_numpy())
    counts = np.argmax(np.bincount(tokenized_hue))

    key = trailers_row['Place of act'].keys()[0]

    scenes_array = [
        info_df['brightness'].to_numpy(),
        info_df['colorfulness'].to_numpy(),
        info_df['energy'].to_numpy(),
        info_df['tempo'].to_numpy(),
        info_df['amplitude'].to_numpy(),
        info_df['mfcc'].to_numpy(),
    ]
    scenes_array = np.nan_to_num(scenes_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = MinMaxScaler((-1, 1))
    scenes_array = scaler.fit_transform(scenes_array)

    if mean:
        new_scenes_array = []
        for i in range(len(scenes_array)):
            new_scenes_array.append(np.around(np.mean(scenes_array[i]), 3))

        scenes_array = new_scenes_array

    scenes_array.append(
        np.nan_to_num([trailers_row['imdb_rating'][key]], copy=False, nan=0.0, posinf=0.0, neginf=0.0)[0])
    emotions = json.loads(trailers_row['emotions'][key].replace("\'", "\""))
    scenes_array.append(np.around(emotions["negative"], 3))
    scenes_array.append(np.around(emotions["neutral"], 3))
    scenes_array.append(np.around(emotions["positive"], 3))

    scenes_array.append(counts)

    return scenes_array


def get_scenes_info(movies_id, trailers_df):
    info = {}
    scenes_array = []
    info_folder = "/Volumes/Seagate/natasha-diploma/videoinfo"
    for i in range(0, len(movies_id)):
        info_csv = info_folder + '/' + movies_id[i] + '.csv'
        info_df = pd.read_csv(info_csv, index_col=None, header=0)
        info[movies_id[i]] = getVideoInfo(info_df.iloc[:, 4:],
                                          trailers_df.loc[trailers_df['trailers_name'] == movies_id[i]])
        scenes_array.append(info[movies_id[i]])

    return scenes_array


class AVFeaturesModel(tf.keras.Model):
    def __init__(self, vocab_size, id_embed_size, rnn_units):
        super().__init__(self)
        self.id_embedding = tf.keras.layers.Embedding(vocab_size, id_embed_size)
        self.normalization = tf.keras.layers.BatchNormalization()
        self.concat = tf.keras.layers.Concatenate()
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size, activation=activations.relu)

    def call(self, _input, states=None, return_state=False, training=False):
        ids = self.id_embedding(_input[0])
        features = self.normalization(_input[1])
        x = self.concat([ids, features])
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x)

        if return_state:
            return x, states
        else:
            return x

    def predict(self, x, history, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
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

    tokenized = create_numerate_array(movies_id)
    scenes_array = get_scenes_info(movies_id, trailers_df)

    train_tokenized = tokenized[:4500]
    test_tokenized = tokenized[4500:7460]

    train_scenes = scenes_array[:4500]
    test_scenes = scenes_array[4500:7460]

    ids_ds = tf.data.Dataset.from_tensor_slices(train_tokenized)
    features_ds = tf.data.Dataset.from_tensor_slices(train_scenes)

    ids_ds_test = tf.data.Dataset.from_tensor_slices(test_tokenized)
    features_ds_test = tf.data.Dataset.from_tensor_slices(test_scenes)

    ds_test = tf.data.Dataset.zip((ids_ds_test, features_ds_test))
    ds = tf.data.Dataset.zip((ids_ds, features_ds))
    seq_length = 10

    sequences = ds.batch(seq_length + 1, drop_remainder=True)
    ids_seq = ids_ds.batch(seq_length + 1, drop_remainder=True)
    features_seq = features_ds.batch(seq_length + 1, drop_remainder=True)
    sequences_test = ds_test.batch(seq_length + 1, drop_remainder=True)

    def full_split_input_target(id_sequence, feature_sequence):
        input_text = (id_sequence[:-1], feature_sequence[:-1])
        target_text = (id_sequence[1:])
        return input_text, target_text

    def split_input_target(seq):
        input_text = seq[:-1]
        target_text = seq[1:]
        return input_text, target_text

    ds = sequences.map(full_split_input_target)
    ids_ds = ids_seq.map(split_input_target)
    features_ds = features_seq.map(split_input_target)
    ds_test = sequences_test.map(full_split_input_target)

    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 1
    BUFFER_SIZE = 10000

    ds = (
        ds
            .shuffle(BUFFER_SIZE)
            .batch(TEST_BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

    ids_ds = (
        ids_ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

    features_ds = (
        features_ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )

    ds_test = (
        ds_test
            .shuffle(BUFFER_SIZE)
            .batch(TEST_BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

    ds = sequences.map(full_split_input_target)

    BATCH_SIZE = 1
    BUFFER_SIZE = 10000

    ds = (ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    model = AVFeaturesModel(
        vocab_size=vocab_size,
        id_embed_size=id_embed_size,
        rnn_units=rnn_units)

    for _input, _output in ds.take(1):
        model(_input)

    model.summary()

    return model, ds, ds_test, vocab


def train_model():
    log_file = "/Users/nutochkina/Desktop/VKR/av_features_model/history.ckpt"
    model, ds, ds_test, vocab = setup_model()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_file, save_weights_only=True, verbose=1)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
        name='sparse_categorical_accuracy', dtype=None
    )])

    EPOCHS = 60
    model.fit(ds, validation_data=ds_test, epochs=EPOCHS, callbacks=[cp_callback])


def evaluate_and_predict():
    log_dir = "/Users/nutochkina/Desktop/VKR/av_features_model/"
    model, ds, ds_test, vocab = setup_model()
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None)])

    latest = tf.train.latest_checkpoint(log_dir)
    model.load_weights(latest)
    model.evaluate(ds_test, batch_size=1, return_dict=True)

    for _input, _output in ds_test.take(1):
        values = model.predict(_input, _input[0].numpy())
        print(vocab[_input[0].numpy()[0][-1] - 1])
        print(vocab[values[0] - 1])









