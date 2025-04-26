import json
import networkx as nx
import os
import logging

# Configure logging level (e.g., INFO, WARNING, ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_conversation_graph(data_dir='data', original_data_filename='raw/cmv_usable.jsonl', annotation_filename='annotations/cmv_relations_karthik.json'):
    """
    Builds a directed graph from conversation data, focusing on
    annotated utterances and their direct parents.

    Loads utterance data and annotation data, identifies relevant nodes
    (annotated utterances and their parents), builds a graph with these
    nodes and connecting edges, and applies ASN (attack/support/neutral)
    relation labels to edges based on annotations.

    Args:
        data_dir (str): Root directory containing data files.
        original_data_filename (str): Path to the main utterance data file (relative to data_dir).
        annotation_filename (str): Path to the annotation JSON file (relative to data_dir).

    Returns:
        networkx.DiGraph or None: The constructed graph, or None if critical errors occur.
    """

    # Construct full file paths
    original_data_file = os.path.join(data_dir, original_data_filename)
    annotated_data_file = os.path.join(data_dir, annotation_filename)

    # --- Load Annotation Data ---
    raw_annotations = []
    annotation_ids = set() # Store IDs of annotated utterances
    logging.info(f"Loading annotations from: {annotated_data_file}")
    try:
        with open(annotated_data_file, 'r', encoding='utf-8') as f:
            loaded_annotations = json.load(f)
            if isinstance(loaded_annotations, list):
                 valid_annotations = []
                 for item in loaded_annotations:
                     item_id = item.get('id')
                     if isinstance(item, dict) and item_id is not None:
                         # Ensure IDs are strings for consistency
                         item['id'] = str(item_id)
                         valid_annotations.append(item)
                         annotation_ids.add(item['id'])
                     else:
                         logging.warning(f"Skipping invalid item in annotation data: {item}")
                 raw_annotations = valid_annotations
                 logging.info(f"Loaded {len(raw_annotations)} valid annotations. Found {len(annotation_ids)} unique annotated utterance IDs.")
            else:
                 logging.warning(f"Annotation file {annotated_data_file} is not a JSON list.")
                 raw_annotations = []
    except FileNotFoundError:
        logging.error(f"Annotation file '{annotated_data_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from annotation file: {annotated_data_file}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error reading annotation file {annotated_data_file}: {e}")
        return None

    if not annotation_ids:
        logging.error("No annotation IDs found. Cannot build graph.")
        return None

    # --- Load Original Utterance Data (All) ---
    # Load all utterances into a dictionary for efficient lookups later
    all_utterances_dict = {}
    logging.info(f"Loading all original data from: {original_data_file}")
    try:
        with open(original_data_file, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)
            if isinstance(loaded_json, list):
                for item in loaded_json:
                    item_id = item.get('id')
                    if isinstance(item, dict) and item_id is not None:
                         item['id'] = str(item_id) # Ensure ID is string
                         all_utterances_dict[item['id']] = item
            else:
                logging.error(f"Original data file {original_data_file} is not a JSON list.")
                return None
    except FileNotFoundError:
        logging.error(f"Original data file not found: {original_data_file}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from {original_data_file}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error reading original data file {original_data_file}: {e}")
        return None

    if not all_utterances_dict:
        logging.error("No valid utterance data loaded from original file.")
        return None

    logging.info(f"Loaded {len(all_utterances_dict)} total utterances from {original_data_filename}.")

    # --- Determine Required Nodes for the Graph ---
    # Start with annotated nodes that exist in the loaded utterance data
    required_node_ids = annotation_ids.intersection(all_utterances_dict.keys())
    parent_ids_to_add = set()
    logging.info(f"{len(required_node_ids)} annotated nodes found in the utterance data.")

    # Identify parents of annotated nodes to include them
    for node_id in required_node_ids:
        utterance_data = all_utterances_dict.get(node_id)
        if utterance_data:
            parent_id_raw = utterance_data.get('reply-to')
            if parent_id_raw is not None:
                parent_id = str(parent_id_raw)
                # Include parent only if it exists in the loaded data
                if parent_id in all_utterances_dict:
                    parent_ids_to_add.add(parent_id)

    # Combine annotated nodes and their existing parents
    final_node_ids = required_node_ids.union(parent_ids_to_add)
    logging.info(f"Total nodes to include in the graph (annotated + parents): {len(final_node_ids)}")

    # Filter the main data to include only required nodes
    utterances_for_graph = {node_id: all_utterances_dict[node_id] for node_id in final_node_ids}

    # --- Prepare Annotations for Graph Application ---
    relation_map = {'a': 'attack', 's': 'support', 'n': 'neutral'}
    annotations_for_graph = {}
    invalid_relation_count = 0
    for item in raw_annotations:
         ann_id = item['id']
         # Process annotations only for nodes included in the final graph
         if ann_id in final_node_ids:
             raw_relation = item.get('reply_relation')
             mapped_relation = relation_map.get(raw_relation)
             if mapped_relation:
                 item['reply_relation'] = mapped_relation # Use full relation name
                 annotations_for_graph[ann_id] = item
             else:
                 logging.warning(f"Annotation for '{ann_id}' has unknown relation code: '{raw_relation}'.")
                 invalid_relation_count += 1
                 annotations_for_graph[ann_id] = item # Keep data, but relation is invalid

    logging.info(f"Prepared {len(annotations_for_graph)} annotations for graph nodes. Skipped {invalid_relation_count} invalid relation codes.")

    # --- Build the Graph ---
    G = nx.DiGraph()

    # Add Nodes
    logging.info("Adding required nodes to the graph...")
    for node_id in final_node_ids:
        utt_data = utterances_for_graph[node_id]
        # Prepare node attributes, ensuring basic types
        text = str(utt_data.get('text', ''))
        user = str(utt_data.get('user', ''))
        root = str(utt_data.get('root', ''))
        timestamp = utt_data.get('timestamp')
        score = utt_data.get('score')
        # Check for delta award marker
        delta_awarded = False
        if 'meta' in utt_data and isinstance(utt_data['meta'], dict):
             # Basic check for delta symbol or command in text
             if '\u2206' in text or '!delta' in text.lower():
                 delta_awarded = True
        delta_str = str(delta_awarded) # Store as string for GraphML

        G.add_node(node_id,
                   text=text,
                   user=user,
                   root=root,
                   timestamp=timestamp if timestamp is not None else '',
                   score=score if score is not None else 0,
                   delta=delta_str)
    logging.info(f"Added {G.number_of_nodes()} nodes.")

    # Add Edges
    logging.info("Adding edges between required nodes...")
    edge_count = 0
    for node_id in final_node_ids:
        utt_data = utterances_for_graph[node_id]
        parent_id_raw = utt_data.get('reply-to')
        if parent_id_raw is not None:
            parent_id = str(parent_id_raw)
            # Add edge only if the parent is also included in the graph
            if parent_id in final_node_ids:
                G.add_edge(parent_id, node_id, relation='reply') # Default relation
                edge_count += 1
    logging.info(f"Added {edge_count} edges.")

    # Apply ASN Relation Attributes to Edges
    logging.info("Applying ASN relation attributes...")
    asn_added_count = 0
    asn_missing_edge_count = 0
    for ann_id, ann_data in annotations_for_graph.items():
        relation = ann_data.get('reply_relation')
        # Skip if relation wasn't valid/mapped
        if relation not in ['attack', 'support', 'neutral']:
             continue

        parent_id_raw = utterances_for_graph[ann_id].get('reply-to')
        parent_id = str(parent_id_raw) if parent_id_raw is not None else None

        # Check if the edge exists in the filtered graph
        if parent_id and G.has_edge(parent_id, ann_id):
            # Optional: Text verification between annotation and graph parent text
            reply_to_text_annotation = ann_data.get('reply_to_text')
            parent_node_data = G.nodes[parent_id]
            parent_text_graph = parent_node_data.get('text', '')
            if reply_to_text_annotation and parent_text_graph.strip() != str(reply_to_text_annotation).strip():
                 logging.warning(f"Text mismatch for edge ({parent_id} -> {ann_id}). Applying relation anyway.")

            # Apply the relation attribute
            G.edges[parent_id, ann_id]['relation'] = str(relation)
            asn_added_count += 1
        elif parent_id:
            # Parent exists, but edge doesn't (should be rare with this logic)
            logging.warning(f"Cannot apply ASN: Edge {parent_id} -> {ann_id} missing.")
            asn_missing_edge_count += 1
        else:
            # Annotated node has no parent in the data
             logging.warning(f"Cannot apply ASN: Node {ann_id} has no parent.")
             asn_missing_edge_count += 1

    logging.info(f"Applied ASN relations to {asn_added_count} edges.")
    if asn_missing_edge_count > 0:
        logging.warning(f"Could not apply ASN relation for {asn_missing_edge_count} annotations.")

    logging.info(f"Graph construction complete. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIRECTORY = 'data'
    # Select the primary data source file
    # ORIGINAL_DATA_FILENAME = 'raw/cmv_usable_subset.jsonl'
    ORIGINAL_DATA_FILENAME = 'raw/cmv_usable.jsonl'
    # Annotation file path
    ANNOTATION_FILENAME = 'annotations/cmv_relations_karthik.json'
    # Output file for the graph visualization
    OUTPUT_GRAPHML_FILE = 'conversation_graph_annotated_focus.graphml'

    # --- Build Graph ---
    graph = build_conversation_graph(
        data_dir=DATA_DIRECTORY,
        original_data_filename=ORIGINAL_DATA_FILENAME,
        annotation_filename=ANNOTATION_FILENAME
    )

    # --- Export Graph ---
    if graph:
        logging.info("Graph built successfully.")
        try:
            output_path = OUTPUT_GRAPHML_FILE
            logging.info(f"Exporting graph to GraphML file: {output_path}")
            # Export using networkx function
            nx.write_graphml(graph, output_path, encoding='utf-8', infer_numeric_types=True)
            logging.info(f"Graph successfully exported to {output_path}")
            # Print final confirmation to terminal
            print(f"\nGraph construction complete.")
            print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
            print(f"Graph exported to: {output_path}")
            print("You can open this file with visualization software like Gephi.")

        except Exception as e:
            logging.error(f"Failed to export graph to GraphML: {e}")
            print(f"\nError exporting graph: {e}")

    else:
        logging.error("Graph construction failed. Cannot export.")
        print("\nGraph construction failed.")

    # --- Analysis Example (Commented Out) ---
    # The analysis code previously here can be added back if needed.
    # It would operate on the 'graph' object constructed above.
