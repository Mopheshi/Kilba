import time

from nlp.embeddings_model import load_embeddings
from nlp.nlp_helper import NLPHelper

embeddings_output_pickle = "../files/tensorflow_kilba_word_embeddings.pkl"

if __name__ == '__main__':
    start = time.time()
    embeddings = load_embeddings(embeddings_output_pickle)

    nlp_helper = NLPHelper()
    analogy_word = nlp_helper.find_analogies('hyel', 'yesu', 'tlakəu', embeddings)
    next_words = nlp_helper.next_word_prediction('taɗər alkawal aku nya',
                                                 embeddings)
    correction = nlp_helper.autocorrect('maryam', embeddings)
    end = time.time()

    print(f"Analogy word: {analogy_word}")
    print(f"Next word predictions: {next_words}")
    print(f"Autocorrect suggestions: {correction}")

    print(f"Time taken: {end - start} seconds.")

    # y_true = []
    # y_pred = []
    # y_pred_prob = []
    # classes = [...]
    # evaluate_model(y_true, y_pred, y_pred_prob, classes)
