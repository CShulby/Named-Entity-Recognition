import string


def truecase(token):
    """
    Truecase a token based on its POS tag.

    Args:
        token (spacy.tokens.Token): The token to be truecased.

    Returns:
        str: The truecased token.
    """
    if token.tag_ in ["NN", "NNS"]:
        return token.text.capitalize()
    else:
        return token.text


def preprocess_text(text):
    """
    Preprocess the input text by removing punctuation.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text with punctuation removed.
    """
    # Removing punctuation
    for punctuation_char in string.punctuation:
        text = text.replace(punctuation_char, "")
    return text
