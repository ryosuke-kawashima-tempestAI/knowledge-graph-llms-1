from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pyvis.network import Network

from dotenv import load_dotenv
import os
import asyncio
import re
import datetime

# Load the .env file
load_dotenv()
# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
# api_key = input("Please enter your OpenAI API key: ")
# os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(temperature=0, model_name="gpt-5")

graph_transformer = LLMGraphTransformer(llm=llm)


# Extract graph data from input text
async def extract_graph_data(text):
    """
    Asynchronously extracts graph data from input text using a graph transformer.

    Args:
        text (str): Input text to be processed into graph format.

    Returns:
        list: A list of GraphDocument objects containing nodes and relationships.
    """
    documents = [Document(page_content=text)]
    graph_documents = await graph_transformer.aconvert_to_graph_documents(documents)
    return graph_documents


def visualize_graph(graph_documents):
    """
    Visualizes a knowledge graph using PyVis based on the extracted graph documents.

    Args:
        graph_documents (list): A list of GraphDocument objects with nodes and relationships.

    Returns:
        pyvis.network.Network: The visualized network graph object.
    """
    # Create network
    net = Network(height="1200px", width="100%", directed=True,
                      notebook=False, bgcolor="#222222", font_color="white", filter_menu=True, cdn_resources='remote') 

    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships

    # Build lookup for valid nodes
    node_dict = {node.id: node for node in nodes}
    
    # Filter out invalid edges and collect valid node IDs
    valid_edges = []
    valid_node_ids = set()
    for rel in relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            valid_edges.append(rel)
            valid_node_ids.update([rel.source.id, rel.target.id])

    # Track which nodes are part of any relationship
    connected_node_ids = set()
    for rel in relationships:
        connected_node_ids.add(rel.source.id)
        connected_node_ids.add(rel.target.id)

    # Add valid nodes to the graph
    for node_id in valid_node_ids:
        node = node_dict[node_id]
        try:
            net.add_node(node.id, label=node.id, title=node.type, group=node.type)
        except:
            continue  # Skip node if error occurs

    # Add valid edges to the graph
    for rel in valid_edges:
        try:
            net.add_edge(rel.source.id, rel.target.id, label=rel.type.lower())
        except:
            continue  # Skip edge if error occurs

    # Configure graph layout and physics
    net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
        }
    """)

    output_file = "knowledge_graph.html"
    try:
        net.save_graph(output_file)
        print(f"Graph saved to {os.path.abspath(output_file)}")
        return net
    except Exception as e:
        print(f"Error saving graph: {e}")
        return None


def generate_knowledge_graph(text):
    """
    Generates and visualizes a knowledge graph from input text.

    This function runs the graph extraction asynchronously and then visualizes
    the resulting graph using PyVis.

    Args:
        text (str): Input text to convert into a knowledge graph.

    Returns:
        pyvis.network.Network: The visualized network graph object.
    """
    graph_documents = asyncio.run(extract_graph_data(text))
    # Recorde the generated graph in Mearmaid format
    _export_graph_to_mermaid(graph_documents, filename="./graphs/knowledge_graph.mmd")
    net = visualize_graph(graph_documents)
    return net

## My Added Fuctions Start Here ##
def _export_graph_to_mermaid(graph_documents, direction="TD", include_types=True, filename=None):
    """
    Convert GraphDocument nodes and relationships to a Mermaid diagram string and optionally save it.

    Args:
        graph_documents (list): A list of GraphDocument objects (as returned by the graph transformer).
        direction (str): Mermaid graph direction, e.g. "TD" (top-down), "LR" (left-right). Default "TD".
        include_types (bool): If True, include node types in labels and relationship types as edge labels.
        filename (str or None): If provided, save the mermaid text to this file path.

    Returns:
        str: The generated Mermaid diagram as a string (starts with "graph <direction>").
    
    Example:
        mermaid_text = export_graph_to_mermaid(graph_documents, direction="LR", filename="graph.mmd")
    """
    if not graph_documents:
        return f"graph {direction}\n"

    if filename == None:
        now = datetime.datetime.now()
        filename = f"./knowledge_graph{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}.mmd"

    doc = graph_documents[0]
    nodes = getattr(doc, "nodes", []) or []
    relationships = getattr(doc, "relationships", []) or []

    def _sanitize_id(s):
        # Keep IDs safe for Mermaid (alphanumeric and underscores)
        return re.sub(r"[^A-Za-z0-9_]", "_", str(s))

    def _escape_label(s):
        # Escape double quotes for Mermaid label usage
        return str(s).replace('"', '\\"')

    id_map = {}
    for node in nodes:
        raw_id = getattr(node, "id", None)
        if raw_id is None:
            continue
        safe_id = _sanitize_id(raw_id)
        # Ensure unique safe ids if collisions occur
        suffix = 1
        base = safe_id
        while safe_id in id_map.values():
            suffix += 1
            safe_id = f"{base}_{suffix}"
        id_map[raw_id] = safe_id

    lines = [f"graph {direction}"]

    # Declare nodes (ensures isolated nodes are shown)
    for node in nodes:
        raw_id = getattr(node, "id", None)
        if raw_id is None:
            continue
        safe_id = id_map[raw_id]
        label_parts = [raw_id]
        if include_types:
            node_type = getattr(node, "type", None)
            if node_type:
                label_parts.append(f"({node_type})")
        label = "\\n".join(_escape_label(p) for p in label_parts)
        lines.append(f'{safe_id}["{label}"]')

    # Add edges with optional relationship type labels
    for rel in relationships:
        src_raw = getattr(getattr(rel, "source", None), "id", None)
        tgt_raw = getattr(getattr(rel, "target", None), "id", None)
        if src_raw not in id_map or tgt_raw not in id_map:
            # skip edges that reference unknown nodes
            continue
        src = id_map[src_raw]
        tgt = id_map[tgt_raw]
        rel_label = ""
        if include_types:
            rel_type = getattr(rel, "type", None)
            if rel_type:
                rel_label = f'|{_escape_label(str(rel_type))}|'
        lines.append(f"{src} -->{rel_label} {tgt}")

    mermaid_text = "\n".join(lines) + "\n"

    if filename:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(mermaid_text)
        except Exception:
            # fail silently but still return the text
            pass

    return mermaid_text
