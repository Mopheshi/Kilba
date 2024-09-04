import os
import json
import re


def extract_text(new_text):
    """
    Cleans the input text by removing specific patterns and normalizing whitespace.

    This function performs several cleaning operations on the input text:
    - Removes verse quotations typically found in religious texts.
    - Replaces newline characters with spaces to ensure the text is on a single line.
    - Collapses multiple consecutive spaces into a single space.
    - Removes a specific copyright text at the end of the document.
    - Ensures that words with apostrophes are retained as valid words.

    Parameters:
    - new_text (str): The text to be cleaned.

    Returns:
    - str: The cleaned text, with unnecessary characters and patterns removed.
    """
    try:
        # Retain words with apostrophes like "papa'akur", "tsa'a", "wawa'a"
        # This regular expression allows words with apostrophes to remain intact
        new_text = re.sub(r'\b(\w+\W\w+)\b', lambda x: x.group(), new_text)

        # Remove verse quotations like (Mat 27:1-2, 11-14; Mar 15:1-5; Yoh 18:28-38)
        new_text = re.sub(
            r'\(\w{3,4} \d{1,2}:\d{1,2}(-\d{1,2})?(, \d{1,2}(-\d{1,2})?)*(; \w{3,4} \d{1,2}:\d{1,2}(-\d{1,2})?)*\)', '',
            new_text)

        # Replace new lines with spaces
        new_text = new_text.replace('\n', ' ')

        # Remove multiple spaces
        new_text = re.sub(r'\s+', ' ', new_text)

        # Remove the copyright text at the end
        copyrightText = "© Wycliffe Bible Translators, Inc. and © The Nigeria Bible Translation Trust 2018"
        new_text = new_text.replace(copyrightText, "")

        return new_text.strip()
    except Exception as e:
        print(f"Error in extract_text: {e}")
        return ""


def process_dictionary_content(content):
    """
    Processes the content of a dictionary, extracting and cleaning text.

    This function is specifically designed to handle dictionary structures in JSON files.
    It applies the `extract_text` function to each value in the dictionary.

    Parameters:
    - content (dict): The dictionary content to process.

    Returns:
    - list: A list of cleaned text strings extracted from the dictionary values.
    """
    try:
        return [extract_text(text) for key, text in content.items()]
    except Exception as e:
        print(f"Error in process_dictionary_content: {e}")
        return []


def process_list_content(content):
    """
    Processes the content of a list, extracting and cleaning text.

    This function is specifically designed to handle list structures in JSON files.
    It applies the `extract_text` function to each dictionary item in the list that has a 'newText' key.

    Parameters:
    - content (list): The list content to process.

    Returns:
    - list: A list of cleaned text strings extracted from the list items.
    """
    try:
        return [extract_text(item.get('newText', '')) for item in content if 'newText' in item]
    except Exception as e:
        print(f"Error in process_list_content: {e}")
        return []


def process_file(file_path):
    """
    Reads a JSON file from the given path, processes its content, and returns cleaned text.

    This function determines whether the content is a dictionary or a list and processes it accordingly.

    Parameters:
    - file_path (str): The path to the JSON file to be processed.

    Returns:
    - list: A list of cleaned text strings extracted from the file's content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Check if the content is a dictionary or a list and process accordingly
            if isinstance(data, dict):
                return process_dictionary_content(data)
            elif isinstance(data, list):
                return process_list_content(data)

        return []
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def process_files_in_folder(folder_path, output_file):
    """
    Processes all JSON files in a given folder, concatenates their cleaned text, and saves it to an output file.

    This function finds all JSON files in the specified folder, processes each file to extract and clean text,
    and then concatenates all the cleaned text into a single corpus. This corpus is then saved to the specified
    output file.

    Parameters:
    - folder_path (str): The path to the folder containing JSON files to process.
    - output_file (str): The path to the file where the concatenated corpus should be saved.

    Returns:
    None
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    # Sort files based on the numerical value in their names
    files.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

    if not files:
        print("No JSON files found in the folder.")
        return

    results = []
    for file in files:
        result = process_file(file)
        results.append(result)

    corpus = [text for sublist in results for text in sublist if text]

    if not corpus:
        print("No text was extracted from the files.")
        return

    giant_corpus = " ".join(corpus)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(giant_corpus)

    print(f"Processed {len(files)} files.")
    print(f"The corpus has been written to {output_file}")


def clean_corpus(input_file, output_file):
    """
    Cleans a corpus by removing verse numbers, extra spaces, punctuation, and converting to lowercase.

    This function reads a corpus from an input file, applies several cleaning operations to remove
    verse numbers, normalize spaces, remove punctuation, and convert all text to lowercase. The cleaned
    corpus is then saved to an output file.

    Parameters:
    - input_file (str): The path to the file containing the original corpus.
    - output_file (str): The path to the file where the cleaned corpus will be saved.

    Returns:
    None
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Remove verse numbers within the text (numbers attached to the beginning of words)
        cleaned_text = re.sub(r'\b\d+\w*', '', text)

        # Remove extra spaces created by the removal
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Remove punctuations that don't convey meaning
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

        # Change to lowercase
        cleaned_text = cleaned_text.lower()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        print(f"Cleaned verse numbers and written to {output_file}")
    except Exception as e:
        print(f"Error in clean_corpus: {e}")


if __name__ == "__main__":
    folder_path = "../Kilba Bible Scraps"
    output_file = "../files/kilba_corpus.txt"
    clean_output_file = "../files/clean_kilba_corpus.txt"

    process_files_in_folder(folder_path, output_file)
    clean_corpus(output_file, clean_output_file)
