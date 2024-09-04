import os

import json
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_corpus(file_path):
    """
    Reads a text corpus from a file and returns it as a string.
    :param file_path:
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
    return corpus


def load_embeddings(pickle_path):
    """
    Loads word embeddings from a pickle file.

    Parameters:
    - pickle_path (str): Path to the pickle file containing the word embeddings.

    Returns:
    - dict: A dictionary of word embeddings loaded from the pickle file.
    """
    with open(pickle_path, 'rb') as file:
        word_embeddings = pickle.load(file)
        return word_embeddings


class EmbeddingsModel:
    """
    A class to handle word embeddings operations using TensorFlow and Keras.

    This class provides methods to build vocabulary, create sequences, define and train a word embeddings model,
    and retrieve word embeddings from the trained model.
    """

    def __init__(self, embedding_dim=100, max_length=100):
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    def build_vocabulary(self, corpus, num_words=None):
        """
        Initializes a Keras Tokenizer and fits it on the provided text corpus to build a vocabulary.

        Parameters:
        - corpus (str): The text corpus to tokenize and build the vocabulary from.
        - num_words (int, optional): The maximum number of words to keep in the vocabulary.

        Returns:
        - Tokenizer: A Keras Tokenizer object fitted on the corpus.
        """
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts([corpus])

    def create_sequences(self, corpus):
        """
        Converts the text corpus into sequences of integers using the tokenizer and pads them to a uniform length.

        Parameters:
        - corpus (str): The text corpus to convert into sequences.

        Returns:
        - numpy.ndarray: An array of padded integer sequences representing the text corpus.
        """
        sequences = self.tokenizer.texts_to_sequences([corpus])[0]
        padded_sequences = pad_sequences([sequences], maxlen=self.max_length, padding='post')
        return padded_sequences[0]

    def define_model(self):
        """
        Defines a more complex TensorFlow Keras model with an embedding layer for training word embeddings.

        Returns:
        - tf.keras.Model: A compiled Keras model ready for training.
        """
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size,
                                      output_dim=self.embedding_dim,
                                      input_length=self.max_length),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),  # Increased complexity
            tf.keras.layers.Dropout(0.5),  # Regularization
            tf.keras.layers.Dense(self.embedding_dim, activation='linear')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    def train_model(self, sequences, epochs=1000, batch_size=32):
        """
        Trains the word embeddings model using the provided sequences.

        Parameters:
        - sequences (numpy.ndarray): The input sequences for training.
        - epochs (int): The maximum number of training epochs.
        - batch_size (int): The number of samples per gradient update.
        """
        context_words = sequences[:-1]
        target_words = sequences[1:]

        # Ensure context_words and target_words have the same length
        min_length = min(len(context_words), len(target_words))
        context_words = context_words[:min_length]
        target_words = target_words[:min_length]

        # Reshape to ensure compatibility with the model
        context_words = context_words.reshape((min_length, 1))
        target_words = target_words.reshape((min_length, 1))

        # Define EarlyStopping and LearningRateScheduler callbacks
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * 0.95 ** epoch) # Exponential Learning rate decay

        # Train the model with EarlyStopping and LearningRateScheduler
        self.model.fit(context_words, target_words, epochs=epochs, batch_size=batch_size,
                       callbacks=[early_stopping, lr_scheduler])

    def get_word_embeddings(self):
        """
        Extracts word embeddings from the trained model and returns them in a dictionary.

        Returns:
        - dict: A dictionary mapping words to their embedding vectors.
        """
        embeddings = self.model.layers[0].get_weights()[0]  # Extract embeddings
        word_index = self.tokenizer.word_index
        word_embeddings = {word: embeddings[index] for word, index in word_index.items() if index < len(embeddings)}
        return word_embeddings

    @staticmethod
    def save_embeddings(word_embeddings, pickle_path, json_path):
        """
        Saves the word embeddings to both pickle and JSON files.

        Parameters:
        - word_embeddings (dict): The dictionary of word embeddings to save.
        - pickle_path (str): Path to save the pickle file.
        - Json_path (str): Path to save the JSON file.
        """
        with open(pickle_path, 'wb') as file:
            pickle.dump(word_embeddings, file)
            print(f"Word embeddings saved to {pickle_path}")

        word_embeddings_json = {word: embeddings.tolist() for word, embeddings in word_embeddings.items()}

        with open(json_path, 'w') as file:
            json.dump(word_embeddings_json, file)
            print(f"Word embeddings saved to {json_path}")
