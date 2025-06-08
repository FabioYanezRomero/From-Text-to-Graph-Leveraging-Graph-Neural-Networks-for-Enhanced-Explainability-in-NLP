import networkx as nx
from src.Clean_Code.Tree_Generation.constituency import ConstituencyTreeGenerator

# Test sentence
sentence = "No wit, only labored gags."

generator = ConstituencyTreeGenerator(device="cpu")
graphs = generator.get_graph([sentence])
G = graphs[0]

# Print nodes sorted by id
nodes_sorted = sorted(G.nodes(data=True), key=lambda x: x[1]['id'])
for key, data in nodes_sorted:
    print(f"Node key: {key}, id: {data['id']}, label: {data['label']}")

# Check ids are strictly sequential
ids = [data['id'] for _, data in nodes_sorted]
assert ids == list(range(len(ids))), f"IDs are not strictly sequential: {ids}"
print("IDs are strictly sequential and contiguous.")
