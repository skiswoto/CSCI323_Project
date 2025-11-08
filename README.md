### Setup Guide

##### Prerequisites
Data in data/folder
- USA-road-d.NY.co 
- USA-road-d.NY.gr 
- USA-road-t.NY.gr

1. Clone the repo
```
git clone
```

2. Create a virtual environment
```bash
python -m env myenv
source myenv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Load graphs from data folder
```bash
python load_graphs.py
```

5. Generate queries for input
```bash
python generate_queries.py
```

6. Run Benchmarks
```bash
python run_benchmark.py
```

7. Visualise results
```bash
python visualise_results.py
```

