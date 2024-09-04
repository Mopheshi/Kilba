from nlp.embeddings_model import EmbeddingsModel, read_corpus

clean_corpus_file = '../files/clean_kilba_corpus.txt'
embeddings_output_pickle = '../files/tensorflow_kilba_word_embeddings.pkl'
embeddings_output_json = '../files/tensorflow_kilba_word_embeddings.json'

corpus = read_corpus(clean_corpus_file)

if corpus:
    embeddings_model = EmbeddingsModel(embedding_dim=100, max_length=100)
    embeddings_model.build_vocabulary(corpus)
    sequences = embeddings_model.create_sequences(corpus)
    embeddings_model.define_model()
    embeddings_model.train_model(sequences, epochs=1000, batch_size=32)
    word_embeddings = embeddings_model.get_word_embeddings()
    embeddings_model.save_embeddings(word_embeddings, embeddings_output_pickle, embeddings_output_json)
else:
    print('No corpus available for creating embeddings.')
