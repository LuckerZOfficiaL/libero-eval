import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import action_error as A
pairs = [tuple(x) for x in json.load(open("sweep/_ae_pairs.json"))]
A.generate_labels(pairs, out="results/analysis/action_error_labels.json")
print("AE LABELS DONE", flush=True)
