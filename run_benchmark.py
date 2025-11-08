import pandas as pd
import random
from load_graph import load_dimacs, load_coordinates
from benchmarks import (
    benchmark_dijkstra, 
    benchmark_astar, 
    benchmark_yen,
    benchmark_yen_astar,
    benchmark_psb,
    benchmark_sb
)

def run_fast_benchmark():
    """
    FAST benchmark strategy:
    - Dijkstra/A*: All 150 queries (fast enough)
    - K-shortest: Only 20 SHORT queries (to avoid 30+ min queries)
    
    Total time: ~30 minutes instead of hours
    """
    
    print("="*80)
    print(" FAST K-SHORTEST PATHS BENCHMARK")
    print(" Strategy: Full coverage for baselines, SHORT queries only for K-shortest")
    print("="*80)
    
    # Load graphs and coordinates
    print("\n[1/4] Loading graphs and coordinates...")
    G_dist = load_dimacs('data/USA-road-d.NY.gr')
    G_time = load_dimacs('data/USA-road-t.NY.gr')
    coords = load_coordinates('data/USA-road-d.NY.co')
    
    # Load queries
    print("\n[2/4] Loading queries...")
    queries = pd.read_csv('data/queries.csv')
    print(f"✓ Loaded {len(queries)} queries")
    
    # Select SHORT queries only for K-shortest algorithms
    print("\n[3/4] Selecting query subsets...")
    random.seed(42)
    
    short_queries = queries[queries['bucket'] == 'short']
    k_shortest_queries = short_queries.sample(n=20, random_state=42).index.tolist()
    
    print(f"✓ Dijkstra/A* will test: ALL {len(queries)} queries")
    print(f"✓ K-shortest will test: {len(k_shortest_queries)} SHORT queries only")
    print(f"\n  Rationale:")
    print(f"    - Medium queries: 5-30 min each → Prohibitive")
    print(f"    - Long queries: 30+ min each → Prohibitive")
    print(f"    - Short queries: <10 sec each → Feasible")
    
    # Calculate total runs
    baseline_runs = len(queries) * 2 * 2  # All queries, 2 objectives, 2 algorithms
    k_shortest_runs = len(k_shortest_queries) * 2 * 4 * 5  # SHORT only, 2 obj, 4 algos, 5 K values (3,5,10,15,20)
    total_runs = baseline_runs + k_shortest_runs
    
    print(f"\nTotal benchmark runs: {total_runs}")
    print(f"  - Baseline (Dijkstra/A*): {baseline_runs} runs")
    print(f"  - K-shortest (Yen/Yen-A*/PSB/SB): {k_shortest_runs} runs")
    print(f"    (20 queries × 2 objectives × 4 algorithms × 5 K values)")
    print(f"\nEstimated time: 45-60 minutes")
    
    results = []
    current_run = 0
    
    # ════════════════════════════════════════════════════════════
    # PHASE 1: Baseline algorithms (ALL queries)
    # ════════════════════════════════════════════════════════════
    print("\n[4/4] Running experiments...")
    print("\n" + "─"*80)
    print("PHASE 1: Baseline Algorithms (All Queries)")
    print("─"*80)
    
    for objective, G in [('distance', G_dist), ('time', G_time)]:
        print(f"\nObjective: {objective.upper()}")
        
        for idx, row in queries.iterrows():
            source = row['source']
            target = row['target']
            bucket = row['bucket']
            
            # Dijkstra
            current_run += 1
            print(f"[{current_run}/{total_runs}] {objective[:4]} | {bucket[:3]:3s} | "
                  f"Dijkstra | Q{idx:3d}", end='\r')
            
            dijk_result = benchmark_dijkstra(G, source, target, weight='weight')
            results.append({
                'objective': objective,
                'algorithm': 'Dijkstra',
                'variant': 'baseline',
                'K': 1,
                'query_id': row['query_id'],
                'bucket': bucket,
                'source': source,
                'target': target,
                'distance_km': row['distance_km'],
                'path_edges': len(dijk_result['path']) - 1 if dijk_result['path'] else 0,
                'cost': dijk_result['cost'],
                'nodes_expanded': dijk_result['nodes_expanded'],
                'runtime_sec': dijk_result['runtime_sec'],
                'peak_memory_mb': dijk_result['peak_memory_mb'],
                'num_paths': 1
            })
            
            # A*
            current_run += 1
            print(f"[{current_run}/{total_runs}] {objective[:4]} | {bucket[:3]:3s} | "
                  f"A*       | Q{idx:3d}", end='\r')
            
            astar_result = benchmark_astar(G, source, target, coords, 
                                          weight='weight', objective=objective)
            results.append({
                'objective': objective,
                'algorithm': 'A*',
                'variant': 'baseline',
                'K': 1,
                'query_id': row['query_id'],
                'bucket': bucket,
                'source': source,
                'target': target,
                'distance_km': row['distance_km'],
                'path_edges': len(astar_result['path']) - 1 if astar_result['path'] else 0,
                'cost': astar_result['cost'],
                'nodes_expanded': astar_result['nodes_expanded'],
                'runtime_sec': astar_result['runtime_sec'],
                'peak_memory_mb': astar_result['peak_memory_mb'],
                'num_paths': 1
            })
        
        print()
    
    # ════════════════════════════════════════════════════════════
    # PHASE 2: K-Shortest algorithms (SHORT queries only)
    # ════════════════════════════════════════════════════════════
    print("\n" + "─"*80)
    print("PHASE 2: K-Shortest Path Algorithms (SHORT queries only)")
    print("─"*80)
    
    for objective, G in [('distance', G_dist), ('time', G_time)]:
        print(f"\nObjective: {objective.upper()}")
        
        for idx in k_shortest_queries:
            row = queries.iloc[idx]
            source = row['source']
            target = row['target']
            bucket = row['bucket']
            
            for K in [3, 5, 10, 15, 20]:  # INCREASED K range
                
                # Yen-Dijkstra
                current_run += 1
                print(f"[{current_run}/{total_runs}] {objective[:4]} | short | "
                      f"Yen-D K={K} | Q{idx:3d}", end='\r')
                
                yen_result = benchmark_yen(G, source, target, K=K, weight='weight')
                results.append({
                    'objective': objective,
                    'algorithm': 'Yens',
                    'variant': 'Dijkstra-subroutine',
                    'K': K,
                    'query_id': row['query_id'],
                    'bucket': bucket,
                    'source': source,
                    'target': target,
                    'distance_km': row['distance_km'],
                    'path_edges': yen_result['total_path_edges'] / yen_result['num_paths'] if yen_result['num_paths'] > 0 else 0,
                    'cost': yen_result['best_cost'],
                    'nodes_expanded': 0,
                    'runtime_sec': yen_result['runtime_sec'],
                    'peak_memory_mb': yen_result['peak_memory_mb'],
                    'num_paths': yen_result['num_paths']
                })
                
                # Yen-A*
                current_run += 1
                print(f"[{current_run}/{total_runs}] {objective[:4]} | short | "
                      f"Yen-A K={K} | Q{idx:3d}", end='\r')
                
                yen_astar_result = benchmark_yen_astar(G, source, target, coords, 
                                                       K=K, weight='weight', objective=objective)
                results.append({
                    'objective': objective,
                    'algorithm': 'Yens',
                    'variant': 'Astar-subroutine',
                    'K': K,
                    'query_id': row['query_id'],
                    'bucket': bucket,
                    'source': source,
                    'target': target,
                    'distance_km': row['distance_km'],
                    'path_edges': yen_astar_result['total_path_edges'] / yen_astar_result['num_paths'] if yen_astar_result['num_paths'] > 0 else 0,
                    'cost': yen_astar_result['best_cost'],
                    'nodes_expanded': 0,
                    'runtime_sec': yen_astar_result['runtime_sec'],
                    'peak_memory_mb': yen_astar_result['peak_memory_mb'],
                    'num_paths': yen_astar_result['num_paths']
                })
                
                # PSB
                current_run += 1
                print(f"[{current_run}/{total_runs}] {objective[:4]} | short | "
                      f"PSB   K={K} | Q{idx:3d}", end='\r')
                
                psb_result = benchmark_psb(G, source, target, K=K, weight='weight')
                results.append({
                    'objective': objective,
                    'algorithm': 'PSB',
                    'variant': 'parsimonious-sidetrack',
                    'K': K,
                    'query_id': row['query_id'],
                    'bucket': bucket,
                    'source': source,
                    'target': target,
                    'distance_km': row['distance_km'],
                    'path_edges': psb_result['total_path_edges'] / psb_result['num_paths'] if psb_result['num_paths'] > 0 else 0,
                    'cost': psb_result['best_cost'],
                    'nodes_expanded': 0,
                    'runtime_sec': psb_result['runtime_sec'],
                    'peak_memory_mb': psb_result['peak_memory_mb'],
                    'num_paths': psb_result['num_paths']
                })
                
                # SB
                current_run += 1
                print(f"[{current_run}/{total_runs}] {objective[:4]} | short | "
                      f"SB    K={K} | Q{idx:3d}", end='\r')
                
                sb_result = benchmark_sb(G, source, target, K=K, weight='weight')
                results.append({
                    'objective': objective,
                    'algorithm': 'SB',
                    'variant': 'sidetrack',
                    'K': K,
                    'query_id': row['query_id'],
                    'bucket': bucket,
                    'source': source,
                    'target': target,
                    'distance_km': row['distance_km'],
                    'path_edges': sb_result['total_path_edges'] / sb_result['num_paths'] if sb_result['num_paths'] > 0 else 0,
                    'cost': sb_result['best_cost'],
                    'nodes_expanded': 0,
                    'runtime_sec': sb_result['runtime_sec'],
                    'peak_memory_mb': sb_result['peak_memory_mb'],
                    'num_paths': sb_result['num_paths']
                })
        
        print()
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    df_results = pd.DataFrame(results)
    output_file = 'benchmark_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"✅ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    
    for obj in ['distance', 'time']:
        print(f"\n{obj.upper()}:")
        subset = df_results[df_results['objective'] == obj]
        
        # Baseline
        for alg in ['Dijkstra', 'A*']:
            data = subset[subset['algorithm'] == alg]
            if len(data) > 0:
                print(f"  {alg}: {data['runtime_sec'].mean():.4f}s avg "
                      f"({len(data)} queries)")
        
        # K-shortest
        for alg in ['Yens', 'PSB', 'SB']:
            data = subset[(subset['algorithm'] == alg) & (subset['K'] == 3)]
            if len(data) > 0:
                print(f"  {alg} K=3: {data['runtime_sec'].mean():.4f}s avg "
                      f"({len(data)} queries)")
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    run_fast_benchmark()