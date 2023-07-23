import torch
import spacy
from transformers import RobertaTokenizer, RobertaModel
from preprocess import truecase


def extract_relations(roberta_model, entities, truecased_text):
    """
    Extract relations between entities using RoBERTa embeddings.

    Args:
        roberta_model (RobertaModel): Pretrained RoBERTa model.
        entities (list): List of tuples containing named entities and their types.
        truecased_text (str): Input text containing the truecased entities.

    Returns:
        list: List of tuples containing entity pairs and their relation scores.
    """
    relations = []

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            entity1, _ = entities[i]
            entity2, _ = entities[j]

            # Get the starting and ending positions of each entity in the truecased text
            e1_start = truecased_text.find(entity1)
            e1_end = e1_start + len(entity1)
            e2_start = truecased_text.find(entity2)
            e2_end = e2_start + len(entity2)

            # Replace the entities with markers <e1> and <e2> to indicate their positions in the input
            marked_text = (
                truecased_text[:e1_start]
                + "<e1>"
                + truecased_text[e1_start:e1_end]
                + "</e1>"
                + truecased_text[e1_end:e2_start]
                + "<e2>"
                + truecased_text[e2_start:e2_end]
                + "</e2>"
                + truecased_text[e2_end:]
            )

            # Convert the marked text to RoBERTa input format
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            inputs = tokenizer.encode_plus(
                marked_text,
                add_special_tokens=True,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                padding="max_length",
                max_length=512,
                truncation=True,
            )

            # Pass the inputs through the RoBERTa model
            with torch.no_grad():
                outputs = roberta_model(**inputs)

            # Extract the entity representations from RoBERTa's output
            e1_representation = outputs.last_hidden_state[0, e1_start : e1_end + 1, :]
            e2_representation = outputs.last_hidden_state[0, e2_start : e2_end + 1, :]

            # Compute the relation score between the entities (you can use a classifier or other methods here)
            relation_score = torch.cosine_similarity(
                e1_representation.mean(dim=0), e2_representation.mean(dim=0), dim=0
            )

            # Round the relation_score to two decimal points
            relation_score = round(relation_score.item(), 2)

            # Add the relation to the relations list if the relation score is above 0.9
            if relation_score > 0.9:
                relations.append((entity1, relation_score, entity2))

    return relations


def create_pipeline():
    """
    Create a spaCy language pipeline for named entity extraction.

    Returns:
        spacy.Language: A spaCy language pipeline.
    """
    nlp = spacy.load("en_core_web_sm")
    return nlp
