import json
import re


def clean_bible_json(input_json):
    """
    Cleans the JSON data of a Bible text by processing content after the 5th newline,
    removing verse numbers, citations, and unwanted footers, while reducing spaces
    and converting text to lowercase.

    Parameters:
    input_json (dict or list): JSON object containing Bible data, either as a dictionary
                               or a list of chapter objects.

    Returns:
    dict: A cleaned version of the input JSON with processed text.
    """
    cleaned_data = {}

    # Check if input is a list or a dictionary
    if isinstance(input_json, list):
        # Process a list of chapters
        for item in input_json:
            chapter = item.get("chapter")  # Adjust the key as per your JSON structure
            content = item.get("content")  # Adjust the key as per your JSON structure

            if chapter and content:
                # Process the content (after 5th newline and other cleaning steps)
                cleaned_data[chapter] = process_content(content)

    elif isinstance(input_json, dict):
        # Process a dictionary of chapters
        for chapter, content in input_json.items():
            # Process the content (after 5th newline and other cleaning steps)
            cleaned_data[chapter] = process_content(content)

    return cleaned_data


def process_content(content):
    """
    Cleans the content by removing unwanted lines, citations, footnotes, and
    verse numbers. Also reduces spaces and converts text to lowercase.
    """
    # Split content into lines and process only after the 5th newline
    lines = content.splitlines()
    if len(lines) > 5:
        content = '\n'.join(lines[5:])
    else:
        content = ''  # If there are not enough lines, set content to empty

    # Remove verse numbers
    content = re.sub(r'\b\d+[A-Za-z]*\b', '', content)

    # Remove citations or footers
    content = re.sub(r'© Wycliffe Bible Translators, Inc. and © The Nigeria Bible Translation Trust 2018', '', content, flags=re.IGNORECASE)

    # Reduce spaces, tabs, or newlines to a single space
    content = re.sub(r'\s+', ' ', content)

    # Convert to lowercase
    return content.strip().lower()


# Example Usage
with open('../Kilba Bible Scraps/1.json', 'r', encoding='utf-8') as file:
    bible_data = json.load(file)

cleaned_bible_data = clean_bible_json(bible_data)

# Save the cleaned data to a new file
with open('cleaned_kilba_bible.json', 'w', encoding='utf-8') as file:
    json.dump(cleaned_bible_data, file, ensure_ascii=False, indent=2)
