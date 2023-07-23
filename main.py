import argparse
from preprocess import preprocess_text
from named_entity_extraction import extract_named_entities
from relation_extraction import extract_relations, create_pipeline
from knowledge_graph import generate_knowledge_graph, plot_knowledge_graph
from transformers import RobertaTokenizer, RobertaModel


def main():
    """
    Main function to generate the knowledge graph from the input text corpus and the gazetteer file.

    Command-line Arguments:
        --corpus (str): Path to the input text corpus.
        --gazetteer (str): Path to the gazetteer file.
    """
    parser = argparse.ArgumentParser(description="Knowledge Graph Generation")
    parser.add_argument("--corpus", type=str, help="Path to the input text corpus")
    parser.add_argument("--gazetteer", type=str, help="Path to the gazetteer file")

    args = parser.parse_args()

    # Load the text corpus from the file
    with open(args.corpus, "r") as corpus_file:
        text = corpus_file.read()

    # Preprocess the text (remove punctuation, etc.)
    preprocessed_text = preprocess_text(text)

    # Extract named entities from the entire text using spaCy
    nlp = create_pipeline()
    entities, truecased_text = extract_named_entities(nlp, text, args.gazetteer)

    # Extract relations between entities using the entire text and the RoBERTa model
    roberta_model = RobertaModel.from_pretrained("roberta-base")
    relations = extract_relations(roberta_model, entities, truecased_text)

    # Generate the knowledge graph
    knowledge_graph = generate_knowledge_graph(entities, relations)

    # Plot the knowledge graph
    plot_knowledge_graph(knowledge_graph)


if __name__ == "__main__":
    main()
