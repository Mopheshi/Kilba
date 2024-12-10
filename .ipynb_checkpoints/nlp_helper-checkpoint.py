import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NLPHelper:
    """
    A class providing NLP utility functions like cosine similarity, analogies, next-word prediction, and autocorrect.

    This class operates on word embeddings provided by the EmbeddingsModel class.
    """

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Computes the cosine similarity between two embedding vectors.

        Parameters:
        - embedding1 (numpy.ndarray): The first embedding vector.
        - embedding2 (numpy.ndarray): The second embedding vector.

        Returns:
        - float: The cosine similarity between the two vectors.
        """
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        cos_sim = np.dot(embedding1, embedding2)
        return cos_sim

    @staticmethod
    def find_analogies(word_a, word_b, word_c, word_embs):
        """
        Finds the analogy word for a given set of three words using their embeddings.

        Example: "man is to king as woman is to ?" -> find_analogies("man", "king", "woman", word_embs)

        Parameters:
        - word_a (str): The first word in the analogy.
        - word_b (str): The second word in the analogy.
        - word_c (str): The third word in the analogy.
        - word_embs (dict): A dictionary of word embeddings.

        Returns:
        - str: The word that completes the analogy.
        """
        try:
            analogy_vector = word_embs[word_b] - word_embs[word_a] + word_embs[word_c]
            best_word = None
            best_similarity = -1
            for word, embedding in word_embs.items():
                similarity = NLPHelper.cosine_similarity(analogy_vector, embedding)
                if similarity > best_similarity and word not in [word_a, word_b, word_c]:
                    best_word = word
                    best_similarity = similarity
            return best_word
        except KeyError as e:
            print(f"Word {e} not found in embeddings.")
            return None

    @staticmethod
    def next_word_prediction(context, word_embeds, top_k=5):
        """
        Predicts the next word given a context based on word embeddings.

        Parameters:
        - context (str): The context string (a sentence or phrase).
        - word_embs (dict): A dictionary of word embeddings.
        - top_k (int): The number of top predictions to return.

        Returns:
        - list: A list of the top predicted words.
        """
        context_vector = np.mean([word_embeds[word] for word in context.split() if word in word_embeds], axis=0)
        similarities = {word: NLPHelper.cosine_similarity(context_vector, emb) for word, emb in word_embeds.items()}
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [word for word, _ in sorted_words]

    @staticmethod
    def autocorrect(word, word_embs, top_k=3):
        """
        Suggests corrections for a given word based on word embeddings.

        Parameters:
        - word (str): The word to correct.
        - word_embs (dict): A dictionary of word embeddings.
        - top_k (int): The number of top suggestions to return.

        Returns:
        - list: A list of the top corrected words.
        """
        if word in word_embs:
            return [word]
        similarities = {w: NLPHelper.cosine_similarity(word_embs[w], np.mean(list(word_embs.values()), axis=0)) for w in
                        word_embs}
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [w for w, _ in sorted_words]
