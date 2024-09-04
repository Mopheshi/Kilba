import numpy as np


def read_corpus(file_path):
    """
    Reads the entire content of a file specified by the file path.

    This function attempts to open and read the content of a file, returning the content as a string.
    If an error occurs during file reading, it prints the error message and returns an empty string.

    Parameters:
    - file_path (str): The path to the file to be read.

    Returns:
    - str: The content of the file as a string or an empty string in case of an error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def tokenize(text):
    """
    Splits the input text into a list of words.

    This function tokenizes the input text by splitting it on whitespace, resulting in a list of words.

    Parameters:
    - text (str): The text to be tokenized.

    Returns:
    - list: A list of words extracted from the input text.
    """
    return text.split()


def build_vocabulary(tokens):
    """
    Builds a vocabulary from a list of tokens.

    This function creates a vocabulary set from the input list of tokens. It also generates two dictionaries
    mapping words to indices and indices to words, respectively, based on the vocabulary.

    Parameters:
    - tokens (list): A list of tokens from which to build the vocabulary.

    Returns:
    - tuple: A tuple containing the vocabulary list, a word-to-index dictionary, and an index-to-word dictionary.
    """
    vocabulary = list(set(tokens))
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    index_to_word = {index: word for index, word in enumerate(vocabulary)}
    return vocabulary, word_to_index, index_to_word


def create_embeddings(vocabulary, embedding_dim=100):
    """
    Creates random embeddings for each word in the vocabulary.

    This function generates a random embedding vector of a specified dimension for each word in the vocabulary.
    The embeddings are stored in a NumPy array with shape (vocab_size, embedding_dim).

    Parameters:
    - vocabulary (list): A list of unique words representing the vocabulary.
    - embedding_dim (int, optional): The dimension of the embedding vectors. Defaults to 100.

    Returns:
    - np.ndarray: A NumPy array containing the embedding vectors for the vocabulary.
    """
    vocab_size = len(vocabulary)
    embeddings = np.random.rand(vocab_size, embedding_dim)
    return embeddings


def save_embeddings(embeddings, word_to_index, output_file):
    """
    Saves the word embeddings to a specified file.

    This function writes the word embeddings to a text file, with each line containing a word followed by its
    embedding vector, separated by spaces. If an error occurs during file writing, it prints the error message.

    Parameters:
    - embeddings (np.ndarray): The embedding vectors for each word in the vocabulary.
    - word_to_index (dict): A dictionary mapping words to their indices in the embedding matrix.
    - output_file (str): The path to the file where the embeddings should be saved.

    Returns:
    None
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for word, index in word_to_index.items():
                embedding_str = " ".join(map(str, embeddings[index]))
                f.write(f"{word} {embedding_str}\n")
        print(f"Word embeddings saved to {output_file}")
    except Exception as e:
        print(f"Error saving embeddings to file {output_file}: {e}")


if __name__ == "__main__":
    clean_corpus_file = "../files/clean_kilba_corpus.txt"
    embeddings_output_file = "../files/basic_kilba_word_embeddings.txt"

    # Step 1: Read and clean the corpus
    corpus = read_corpus(clean_corpus_file)
    if not corpus:
        print("No corpus available for creating embeddings.")
    else:
        # Step 2: Tokenize the text
        tokens = tokenize(corpus)

        # Step 3: Build vocabulary
        vocabulary, word_to_index, index_to_word = build_vocabulary(tokens)

        # Step 4: Create word embeddings
        embedding_dim = 100  # Example dimension size, can be adjusted
        embeddings = create_embeddings(vocabulary, embedding_dim)

        # Step 5: Save embeddings to file
        save_embeddings(embeddings, word_to_index, embeddings_output_file)
