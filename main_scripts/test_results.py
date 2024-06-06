import os
import sys
import torch
import argparse
from datetime import datetime

# Change working directory to project's main directory and add it to path for library and config usage
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from models.set_to_graph import SetToGraph
from performance_eval.eval_test_jets import eval_jets_on_test_set

def parse_args():
    """Define and retrieve command line arguments"""
    argparser = argparse.ArgumentParser(description="Load and evaluate a pre-trained SetToGraph model.")
    argparser.add_argument('--model_path', required=True, help='Path to the pre-trained model file')
    return argparser.parse_args()

def main():
    start_time = datetime.now()

    # Parse command line arguments
    config = parse_args()

    # Initialize a new instance of the SetToGraph model with the same parameters used for training
    model_params = {
        'in_features': 10,
        'out_features': 1,
        'set_fn_feats': [256, 256, 256, 256, 5],
        'method': 'lin2',  # Replace with the actual method used
        'hidden_mlp': [256],
        'predict_diagonal': False,
        'attention': True,
        'set_model_type': 'deepset'
    }

    new_model = SetToGraph(**model_params)

    # Load the weights into the new model instance
    new_model.load_state_dict(torch.load(config.model_path, map_location=torch.device('cpu')))

    # Ensure the model is in evaluation mode
    new_model.eval()

    # Evaluate the model on the test set
    test_results = eval_jets_on_test_set(new_model)

    # Print and save the test results
    print('Test results:')
    print(test_results)

    # Save the test results to a CSV file in the same directory as the model
    results_path = os.path.join(os.path.dirname(config.model_path), "test_results.csv")
    test_results.to_csv(results_path, index=True)

    # Print the total runtime
    print(f'Total runtime: {str(datetime.now() - start_time).split(".")[0]}')

if __name__ == '__main__':
    main()
