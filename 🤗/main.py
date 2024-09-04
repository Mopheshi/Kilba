from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# generator = pipeline("text-generation", model='distilgpt2')
#
# result = generator("In this tutorial, we will learn how to use transformers to", max_length=30, num_return_sequences=2)
# print(result)

# classifier = pipeline("zero-shot-classification")
# result = classifier("I can't wait to see you again...", candidate_labels=["finance", "romance", "travel"])
# print(result)

# Default model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)
# result = classifier("I am not a big fan of boxing...")
# print(result)

sequence = "Using transformers is easy!"
result = tokenizer(sequence)
print(result)
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(sequence)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)

if __name__ == "__main__":
    pass
