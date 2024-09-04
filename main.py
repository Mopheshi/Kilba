import json
import os
import re

kilba_corpus = 'files/kilba_corpus.txt'
cleaned_kilba_corpus = 'files/clean_kilba_corpus.txt'


def preprocess(text_file=kilba_corpus, output_file=cleaned_kilba_corpus):
    try:
        with open(text_file, 'r', encoding='UTF-8') as f:
            content = f.read()

        # Remove the copyright text at the end
        copyright_text = "© Wycliffe Bible Translators, Inc. and © The Nigeria Bible Translation Trust 2018"
        text = content.replace(copyright_text, "")

        # Remove verse quotations like (Mat 27:1-2, 11-14; Mar 15:1-5; Yoh 18:28-38)
        text = re.sub(
            r'\(\w{3,4} \d{1,2}:\d{1,2}(-\d{1,2})?(, \d{1,2}(-\d{1,2})?)*(; \w{3,4} \d{1,2}:\d{1,2}(-\d{1,2})?)*\)',
            '',
            text)

        # Replace new lines with spaces
        text = text.replace('\n', ' ')

        # Remove verse numbers within the text (numbers attached to the beginning of words)
        text = re.sub(r'\b\d+\w*', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Change to lowercase and strip leading/trailing spaces
        text = text.lower().strip()

        # Remove punctuations that don't convey meaning
        text = re.sub(r'[^\w\s]', '', text)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"Text corpus cleaned and written to {output_file}")
    except Exception as e:
        print(f"Error in preprocess: {e}")


for i in range(1, 6):
    with open(f'Kilba Bible Scraps/{i}.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
        data_str = json.dumps(data)
        decoded_text = data_str.encode('UTF-8').decode('unicode_escape')
        plain_text = decoded_text.replace('{', '').replace('}', ''
                                                           ).replace('[', '').replace(']', ''
                                                                                      ).replace('"', '')

        with open(f'Kilba Bible Scraps/{i}.txt', 'w', encoding='UTF-8') as file:
            file.write(plain_text)

for j in range(6, 12):
    with open(f'Kilba Bible Scraps/{j}.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)

        plain_text = ""

        for dictionary in data:
            second_value = list(dictionary.values())[1]
            plain_text += second_value + " "

        with open(f'Kilba Bible Scraps/{j}.txt', 'w', encoding='UTF-8') as file:
            file.write(plain_text)

with open(kilba_corpus, 'w', encoding='UTF-8') as merged_file:
    for k in range(1, 12):
        with open(f'Kilba Bible Scraps/{k}.txt', 'r', encoding='UTF-8') as f:
            merged_file.write(f.read())

for k in range(1, 12):
    os.remove(f'Kilba Bible Scraps/{k}.txt')

preprocess()
