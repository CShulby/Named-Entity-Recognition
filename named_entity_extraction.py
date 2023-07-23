from preprocess import truecase


def read_gazetteer(file_path):
    """
    Read gazetteer entities from a file and return a list of tuples.

    Args:
        file_path (str): Path to the gazetteer file.

    Returns:
        list: List of tuples containing gazetteer entities and their types.
    """
    gazetteer_entities = []
    with open(file_path, "r") as gazetteer_file:
        for line in gazetteer_file:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                gazetteer_entities.append((parts[0], parts[1]))

    return gazetteer_entities


def extract_named_entities(nlp, text, gazetteer_file=None):
    """
    Extract named entities from the input text using spaCy.

    Args:
        nlp (spacy.Language): A spaCy language pipeline.
        text (str): The input text.
        gazetteer_file (str, optional): Path to the gazetteer file. Defaults to None.

    Returns:
        list: List of tuples containing extracted named entities and their types.
    """
    # Preprocess the text using truecase
    doc = nlp(text)
    truecased_text = " ".join([truecase(token) for token in doc])
    doc = nlp(truecased_text)

    entities = []
    previous_label = None
    previous_entity = ""

    for entity in doc.ents:
        # Split multilabels and take the main one
        entity_type = (
            entity.label_.split("-")[1] if "-" in entity.label_ else entity.label_
        )
        # Handle multi-word names
        # Check if the entity type is 'PERSON'
        if entity_type == "PERSON":
            # If the previous entity was also of type 'PERSON', append the current entity text to it
            if previous_label == entity_type:
                previous_entity += " " + entity.text
            else:
                # If the previous entity was not of type 'PERSON', add it to the list if it exists
                if previous_entity:
                    entities.append((previous_entity, previous_label))
                # Set the current entity as the new previous entity
                previous_entity = entity.text
                previous_label = entity_type
        else:
            # If the previous entity was of type 'PERSON', add it to the list if it exists
            if previous_entity:
                entities.append((previous_entity, previous_label))
            # Add the current entity to the list
            entities.append((entity.text, entity_type))
            # Reset the previous_entity and previous_label since the current entity is of a different type
            previous_entity = ""
            previous_label = None
    # Check if there is any remaining previous_entity and add it to the list
    if previous_entity:
        entities.append((previous_entity, previous_label))

    # Add gazetteer entities
    if gazetteer_file:
        gazetteer_entities = read_gazetteer(gazetteer_file)
        for gazetteer_entity, gazetteer_type in gazetteer_entities:
            found = False
            for i, (entity_text, entity_type) in enumerate(entities):
                if gazetteer_entity.lower() == entity_text.lower():
                    entity_type = (
                        gazetteer_type  # Update the entity type to gazetteer type
                    )
                    entities[i] = (gazetteer_entity, entity_type)
                    found = True
                    break
            if not found and gazetteer_entity.lower() in text.lower():
                entities.append((gazetteer_entity, gazetteer_type))

    return entities, truecased_text
