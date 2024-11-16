import json
import os
from typing import List, Dict
import glob

def load_results(results_dir: str = "./results") -> List[Dict]:
    """Load results from all experiment runs."""
    results_file = os.path.join(results_dir, "experiment_results.json")
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found at {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def print_f1_scores(results: List[Dict]):
    """Print F1 scores for each experiment in a formatted table."""
    # Sort results by partial label fraction
    sorted_results = sorted(results, key=lambda x: x['partial_label_fraction'])
    
    # Print header
    print("\nF1 Scores for Different Partial Label Fractions")
    print("-" * 50)
    print(f"{'Partial Label %':>15} | {'F1 Score':>10} | {'Precision':>10} | {'Recall':>10}")
    print("-" * 50)
    
    # Print results
    for result in sorted_results:
        partial_frac = result['partial_label_fraction']
        f1 = result['eval_f1']
        precision = result['eval_precision']
        recall = result['eval_recall']
        
        print(f"{partial_frac:>14.1%} | {f1:>10.4f} | {precision:>10.4f} | {recall:>10.4f}")
    
    print("-" * 50)

def main():
    try:
        # Load and analyze results
        results = load_results()
        print_f1_scores(results)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run the training experiments first.")
    except KeyError as e:
        print(f"Error: Missing key in results file: {e}")
        print("The results file might be corrupted or in an unexpected format.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
