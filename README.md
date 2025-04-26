# Persuasion Modeling 

## Phase 1: Conversation Graph Construction

This project focuses on the first phase of persuasion modeling: building directed graphs representing conversation threads from the Reddit ChangeMyView (CMV) dataset. It specifically uses utterances that have been manually annotated with Attack/Support/Neutral (ASN) relations, [done in this reository]{https://github.com/karthiksairam01/cmv_annotation}. The output is a GraphML file suitable for visualization and analysis in tools (I'm using Gephi)

## Tree Specifications

The goal of this phase is to transform annotated conversational data into a graph structure where:
* Nodes represent individual utterances (comments).
* Directed edges represent the reply-to relationship between utterances.
* Edges corresponding to annotated replies are labeled with their ASN relation ('attack', 'support', 'neutral').

This structure facilitates the analysis of conversational dynamics and persuasion strategies based on the annotated relationships (to be done).

## Data

This script expects data derived from the Convokit CMV dataset.

1.  **Utterance Data:**
    * **File:** Expected in `data/raw/`.
    * **Format:** A **JSON array**, where each object represents an utterance and contains at least:
        * `id`: Unique identifier for the utterance (string).
        * `text`: The text content of the utterance (string).
        * `reply-to`: The ID of the utterance this one is replying to (string), or `null` for Original Posts (OPs).
        * `root`: The ID of the OP of the thread this utterance belongs to (string).
        * Other fields like `user`, `timestamp`, `score`, `meta` may also be present.

2.  **Annotation Data:**
    * **File:** Expected in `data/annotations/` (e.g., `cmv_relations_[name].json`).
    * **Format:** A **JSON array**, where each object represents an annotation for a *replying* utterance and contains:
        * `id`: The ID of the utterance being annotated (the reply) (string).
        * `reply_relation`: The annotated relation of this utterance to its parent ('a' for attack, 's' for support, 'n' for neutral).
        * `reply_to_text`: The text of the parent utterance being replied to (string). Used for verification.
        * `text`: The text of the utterance being annotated (optional but included in the example).

## Script: `tree_builder.py`

This Python script performs the graph construction, and exports the resulting graph to a GraphML file.


## Usage

1.  **Configure Paths:** make sure the data files are available in the data folder.

2.  **Run the tree builder Python file:**
    ```bash
    python tree-init/tree_builder.py
    ```

3.  **Output:** A GraphML file will be created in the root directory.

## Visualization (Gephi)

The generated `.graphml` file can be opened in graph visualization software like [Gephi](https://gephi.org/) to inspect the constructed graph.

