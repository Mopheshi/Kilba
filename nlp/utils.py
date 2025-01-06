import json


def jsonFileToString(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return '\n\n'.join(json.dumps(item) for item in data)
        else:
            return data


def jsonToString(jsfon):
    return json.dumps(json, indent=4, sort_keys=True)


def stringToJson(string):
    return json.loads(string)


def txtToString(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


def stringToTxt(string, file):
    with open(file, 'w') as f:
        f.write(string)


# print(jsonFileToString('../files/kilba_bible.json'))
# print(txtToString('../files/english_bible.txt'))

path = "../files/clean_kilba_corpus.txt"

with open(path, 'r', encoding='utf-8') as f:
    data = f.read()
    print(f"Number of words: {len(data.split())}")
    print(f"Number of characters: {len(data)}")
    print(f"Number of unique words: {len(set(data.split()))}")
    print(f"Number of unique characters: {len(set(data))}")
    print(f"Unique characters: {set(data)}")
