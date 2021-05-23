import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
from get_features.tokenizer.tokenizer import create_numerate_array

tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel('ERROR')


class GruModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def train_model():
    tf.get_logger().setLevel('ERROR')
    sessions_csv = "/Users/nutochkina/Desktop/VKR/sessions1.csv"
    log_file = "/Users/nutochkina/Desktop/VKR/gru_model/history.ckpt"
    sessions_df = pd.read_csv(sessions_csv, index_col=None, header=0)
    sessions_df.drop(sessions_df.columns[sessions_df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
    movies_id = sessions_df.to_numpy()
    movies_id = np.array(movies_id).flatten()
    vocab = np.unique(movies_id)

    tokenized = create_numerate_array(movies_id)
    dataset = tf.data.Dataset.from_tensor_slices(tokenized[:2500])
    sequences = dataset.batch(11, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = (
        dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

    vocab_size = len(vocab)
    embedding_dim = 1024
    rnn_units = 512
    model = GruModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_file,
                                                     save_weights_only=True,
                                                     verbose=1)

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None)])

    EPOCHS = 70
    model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])


def evaluate_and_predict():
    sessions_csv = "/Users/nutochkina/Desktop/VKR/sessions1.csv"
    sessions_df = pd.read_csv(sessions_csv, index_col=None, header=0)
    sessions_df.drop(sessions_df.columns[sessions_df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
    movies_id = sessions_df.to_numpy()
    movies_id = np.array(movies_id).flatten()
    vocab = np.unique(movies_id)
    vocab_size = len(vocab)
    embedding_dim = 1024
    rnn_units = 512
    model = GruModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    tokenized = create_numerate_array(movies_id)
    train = tf.data.Dataset.from_tensor_slices(tokenized[2500:3730])
    sequences = train.batch(11, drop_remainder=True)
    train = sequences.map(split_input_target)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    train = (train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    for input_example_batch, target_example_batch in train.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None)])

    latest = tf.train.latest_checkpoint("/Users/nutochkina/Desktop/VKR/gru_model/")
    model.load_weights(latest)

    model.evaluate(train, batch_size=64, return_dict=True)
    prediction = model.predict([205])
    print(np.argmax(prediction[0]))






