import random
import pandas as pd
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from load_graph import load_dimacs, load_coordinates

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate geographic distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in meters
    """
    R = 6371000.0  # Earth radius in meters
    
    # Convert to radians
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    
    # Haversine formula
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c  # Distance in meters


def generate_bucketed_queries(G, coords, target_per_bucket=50, max_attempts=50000):
    """
    Generate queries classified by geographic distance.
    
    Strategy: For short/medium queries, pick source then search nearby nodes.
    For long queries, pick random pairs.
    
    Buckets:
        - Short: 0-5 km
        - Medium: 5-50 km
        - Long: >50 km
    """
    
    # Get nodes that have coordinates
    nodes_with_coords = [n for n in G.nodes() if n in coords]
    print(f"Nodes with coordinates: {len(nodes_with_coords)}/{G.number_of_nodes()}")
    
    # Initialize buckets
    queries = {
        'short': [],
        'medium': [],
        'long': []
    }
    
    random.seed(42)
    
    print(f"\nGenerating {target_per_bucket} queries per bucket...")
    print("="*70)
    
    # ────────────────────────────────────────────────────────────
    # Phase 1: Generate SHORT queries (0-5 km)
    # ────────────────────────────────────────────────────────────
    print("\n[1/3] Generating SHORT queries (0-5 km)...")
    attempts = 0
    
    while len(queries['short']) < target_per_bucket and attempts < max_attempts:
        attempts += 1
        
        # Pick a random source
        source = random.choice(nodes_with_coords)
        source_lat, source_lon = coords[source]
        
        # Pick another random node
        target = random.choice(nodes_with_coords)
        
        if source == target:
            continue
        
        # Calculate distance
        target_lat, target_lon = coords[target]
        distance_m = haversine(source_lat, source_lon, target_lat, target_lon)
        distance_km = distance_m / 1000.0
        
        # Check if it's in SHORT range
        if distance_km < 5:
            # Check if path exists
            if nx.has_path(G, source, target):
                queries['short'].append({
                    'source': source,
                    'target': target,
                    'distance_km': round(distance_km, 2),
                    'bucket': 'short'
                })
                
                if len(queries['short']) % 10 == 0:
                    print(f"  Progress: {len(queries['short'])}/50", end='\r')
    
    print(f"  ✓ Generated {len(queries['short'])} short queries (attempts: {attempts})")
    
    # ────────────────────────────────────────────────────────────
    # Phase 2: Generate MEDIUM queries (5-50 km)
    # ────────────────────────────────────────────────────────────
    print("\n[2/3] Generating MEDIUM queries (5-50 km)...")
    attempts = 0
    
    while len(queries['medium']) < target_per_bucket and attempts < max_attempts:
        attempts += 1
        
        source = random.choice(nodes_with_coords)
        target = random.choice(nodes_with_coords)
        
        if source == target:
            continue
        
        source_lat, source_lon = coords[source]
        target_lat, target_lon = coords[target]
        distance_m = haversine(source_lat, source_lon, target_lat, target_lon)
        distance_km = distance_m / 1000.0
        
        # Check if it's in MEDIUM range
        if 5 <= distance_km <= 50:
            if nx.has_path(G, source, target):
                queries['medium'].append({
                    'source': source,
                    'target': target,
                    'distance_km': round(distance_km, 2),
                    'bucket': 'medium'
                })
                
                if len(queries['medium']) % 10 == 0:
                    print(f"  Progress: {len(queries['medium'])}/50", end='\r')
    
    print(f"  ✓ Generated {len(queries['medium'])} medium queries (attempts: {attempts})")
    
    # ────────────────────────────────────────────────────────────
    # Phase 3: Generate LONG queries (>50 km)
    # ────────────────────────────────────────────────────────────
    print("\n[3/3] Generating LONG queries (>50 km)...")
    attempts = 0
    
    while len(queries['long']) < target_per_bucket and attempts < max_attempts:
        attempts += 1
        
        source = random.choice(nodes_with_coords)
        target = random.choice(nodes_with_coords)
        
        if source == target:
            continue
        
        source_lat, source_lon = coords[source]
        target_lat, target_lon = coords[target]
        distance_m = haversine(source_lat, source_lon, target_lat, target_lon)
        distance_km = distance_m / 1000.0
        
        # Check if it's in LONG range
        if distance_km > 50:
            if nx.has_path(G, source, target):
                queries['long'].append({
                    'source': source,
                    'target': target,
                    'distance_km': round(distance_km, 2),
                    'bucket': 'long'
                })
                
                if len(queries['long']) % 10 == 0:
                    print(f"  Progress: {len(queries['long'])}/50", end='\r')
    
    print(f"  ✓ Generated {len(queries['long'])} long queries (attempts: {attempts})")
    
    # ────────────────────────────────────────────────────────────
    # Combine and finalize
    # ────────────────────────────────────────────────────────────
    all_queries = queries['short'] + queries['medium'] + queries['long']
    
    # Add sequential query IDs
    for i, q in enumerate(all_queries):
        q['query_id'] = i
    
    # Print summary
    print("\n" + "="*70)
    print(" QUERY GENERATION COMPLETE")
    print("="*70)
    print(f"Total queries generated: {len(all_queries)}")
    print(f"  Short (0-5 km):    {len(queries['short'])} queries")
    print(f"  Medium (5-50 km):  {len(queries['medium'])} queries")
    print(f"  Long (>50 km):     {len(queries['long'])} queries")
    print("="*70 + "\n")
    
    return pd.DataFrame(all_queries)


if __name__ == "__main__":
    # Load graph and coordinates
    G = load_dimacs('data/USA-road-d.NY.gr')
    coords = load_coordinates('data/USA-road-d.NY.co')
    
    # Generate queries
    df = generate_bucketed_queries(G, coords, target_per_bucket=50)
    
    # Save to CSV
    output_file = 'queries.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Queries saved to {output_file}\n")
    
    # Show sample queries from each bucket
    print("Sample queries from each bucket:")
    print(df.groupby('bucket')[['query_id', 'source', 'target', 'distance_km']].head(3))
    
    # Show distance statistics per bucket
    print("\n" + "="*70)
    print("Distance statistics per bucket:")
    print("="*70)
    print(df.groupby('bucket')['distance_km'].describe())