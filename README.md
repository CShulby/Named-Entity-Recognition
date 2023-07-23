# Named Entity Recognition and Knowledge Graph Generation

This package contains scripts for Named Entity Recognition (NER), Relation Extraction (RE) and Knowledge Graph Generation using spaCy, RoBERTa and NetworkX, respectively.

The graph is displayed via Matplotlib. Entities of the same type are displayed on each node with the same color, the colors for each entity type are chosen at random with each run. Relation Scores are displayed as a simple euclidean distance derived from the LLM.

## Installation

1. Clone the repository to your local machine:

```
bash

git clone https://github.com/CShulby/Project_NER.git
```
2. Change to the repository directory:
```
bash

cd Project_NER
```
3. Create a virtual environment (optional but recommended):
```
bash

conda create -n your_env python=3.9
```
4. Activate the virtual environment:
```
bash

    conda activate your_env

    Install the required dependencies:
```

5. Install Requirements (in this exact order)
```    
bash

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
## Usage

Place your input text corpus in a file (e.g., input_corpus.txt) in the root directory of the repository.

Create a gazetteer file (e.g., gazetteer.txt) containing known named entities along with their types. Each line in the gazetteer file should have the tab-delimited format: entity    type    POS (e.g., Leicestershire GPE PNOUN). *Note that the PoS information from the gazatteer is not currently being used, but would be useful if rules were implemented and I have left it for now at the risk of having more information, rather than less. It may be changed it the future.

Run the main script:
```
bash

python main.py --corpus input_corpus.txt (--gazetteer gazetteer.txt)
```

## Extended Description, Notes and Future Work

The script will perform Named Entity Recognition on the input corpus using spaCy, extract relations between entities using RoBERTa, generate a knowledge graph, and visualize it using NetworkX and Matplotlib.

The first time you use it it will download the RoBERTa model which may take a couple of minutes

## Notes

-The NER model is based on spaCy's en_core_web_sm model, which may be updated or changed in the future. Make sure you have the appropriate spaCy model installed.

-The RoBERTa model used in the relation extraction is based on roberta-base from the transformers library. You can change the model if needed.

-The relation extraction function uses cosine similarity to calculate relation scores. This could be changed in the future depending on specific ontologies or requirements.

-The generated knowledge graph will be displayed using Matplotlib. You can save the plot or modify the visualization as needed.
    
-The BERT model is not super efficient (A pretrained LLM was chosen due to time constraints for development). 
    I recommend processing one phrase at a time with this library until some of the future work is completed.
    
## Future Work

-Develop SLAs and contracts for a future service (This may alter the rest of the list)
-Develop testing and training sets
-Create .json dumps for larger corpora
-Use a third party for visualization and queries like GraphQL
-Include parallelization on inference
-Implement a more efficient RE classifier like a CRF trained on data annotated the LLM
-Fine-tune (probably optimizing for recall)
-Implement more robust sentence segmentation, tokenization, normalization and PoS tagging
-For normalization, abbreviations are a top priority
-Spellcheckers, Internet slang, etc.
-More robust truecasing (the simple truecaser implemented from scratch here already showed superior results BERT and other LLMs are generally case sensitive)
-Text Cleaning (non-words, foreign-words, ASCII Art, trash tokens)
-Implement a more robust way to deal with long dependencies and rare-words (the Bane of all LMs hehe)
-Dealing with Multiple, Ambiguous or Fuzzy entity classes
    

