import logging
import networkx as nx

from networkx import Graph, DiGraph
from nltk.corpus import wordnet as wn
from typing import Union, Optional

logger = logging.getLogger(__name__)

try:
    wn.all_synsets
except LookupError as e:
    import nltk
    nltk.download('wordnet')

def build_hypernym_graph(type: str, root: Optional[str] = None) -> DiGraph:
    graph = nx.DiGraph()

    for synset in wn.all_synsets(type):
        graph.add_node(synset.name())
        for hyper in synset.hypernyms():
            graph.add_edge(synset.name(), hyper.name())

    if root is not None:
        graph = get_branch_subgraph(graph, root_name=root)
    return graph
    
def get_branch_subgraph(
    graph: Union[Graph, DiGraph],
    root_name: Union[str,int],
) -> Union[Graph,DiGraph]:
    
    if root_name not in graph:
        raise ValueError(f"{root_name} not in graph")

    # Reverse the graph so we can go downward (hypernym → hyponyms)
    graph_reversed = graph.reverse(copy=False)

    # Get all descendants (i.e., hyponyms of the root)
    descendants = nx.descendants(graph_reversed, root_name)
    descendants.add(root_name)  # include the root itself

    subgraph = graph.subgraph(descendants)
    return subgraph
    
def compute_longest_path_to_children(G):
    # Reverse the graph: edges now point from parent → child
    Gr = G.reverse()

    # Topological order (valid for DAGs)
    topo_order = list(nx.topological_sort(Gr))

    # Store longest path length from node to a leaf
    lp = {node: 0 for node in Gr.nodes}

    # Process in topological order
    for u in reversed(topo_order):  # bottom-up
        for v in Gr.successors(u):  # v is child of u
            lp[u] = max(lp[u], 1 + lp[v])

    return lp