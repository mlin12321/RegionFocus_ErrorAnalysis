import json
import os
import argparse

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


def main(results_file, output_file):
    filtered_results = read_results_metadata(results_file) #read_results_metadata('results/qwen25vl_RegionFocus_7b.json')
    with open(output_file, 'w') as f:
        json.dump(filtered_results, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args.results_file, args.output_file)