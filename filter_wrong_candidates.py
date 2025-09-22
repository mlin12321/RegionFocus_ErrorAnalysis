import json
import os

def read_results_metadata(file_path):
    filtered_results = {}
    filtered_results['details'] = []
    with open(file_path, 'r') as f:
        results = json.load(f)
        for row in results['details']:
            if row['correctness'] == 'wrong':
                filtered_results['details'].append(row)
        filtered_results['metics'] = results['metrics']
        
    return filtered_results


def main():
    filtered_results = read_results_metadata('results/qwen25vl_RegionFocus_7b.json')
    with open('results/qwen25vl_RegionFocus_7b_filtered.json', 'w') as f:
        json.dump(filtered_results, f, indent=4)
    

if __name__ == "__main__":
    main()