import networkx as nx

def load_dimacs(filepath):
    """
    Load a DIMACS .gr file into a NetworkX DiGraph.
    
    Lines starting with 'a' are edges: a source target weight
    """
    G = nx.DiGraph()
    
    print(f"Loading graph from {filepath}...")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Edge: a source target weight
            if line.startswith('a'):
                parts = line.split()
                u = int(parts[1])
                v = int(parts[2])
                w = float(parts[3])
                G.add_edge(u, v, weight=w)
    
    print(f"✓ Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_coordinates(filepath):
    """
    Load node coordinates from DIMACS .co file.
    
    Lines starting with 'v' contain: v node_id x y
    IMPORTANT: Coordinates are scaled by 1,000,000 (integer format)
    """
    coords = {}
    
    print(f"Loading coordinates from {filepath}...")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Coordinate: v node_id x y
            if line.startswith('v'):
                parts = line.split()
                node_id = int(parts[1])
                lon = int(parts[2]) / 1_000_000.0  # Scale down from integer
                lat = int(parts[3]) / 1_000_000.0  # Scale down from integer
                
                # Store as (lat, lon) for easier use with haversine
                coords[node_id] = (lat, lon)
    
    print(f"✓ Loaded coordinates for {len(coords)} nodes")
    return coords


# Test the loader
if __name__ == "__main__":
    G = load_dimacs('data/USA-road-d.NY.gr')
    coords = load_coordinates('data/USA-road-d.NY.co')
    
    # Show first few coordinates
    print(f"\nSample coordinates:")
    for node_id in list(coords.keys())[:5]:
        lat, lon = coords[node_id]
        print(f"  Node {node_id}: lat={lat:.6f}°, lon={lon:.6f}°")