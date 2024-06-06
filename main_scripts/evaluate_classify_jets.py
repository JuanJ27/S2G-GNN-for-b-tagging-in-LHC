import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import uproot3 as uproot

# Change working directory to project's main directory and add it to the path for library and config usage
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from models.classifier import JetClassifier
from dataloaders import jets_loader

DEVICE = 'cpu'

def parse_args():
    """Define and retrieve command line arguments"""
    argparser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    argparser.add_argument('-b', '--bs', default=1000, type=int, help='Batch size to use')
    argparser.add_argument('--vertexing_model_type')
    argparser.add_argument('--path_to_trained_model', default=None, help='Path to trained model')
    argparser.add_argument('--outputfilename')
    argparser.add_argument('--use_rave', dest='use_rave', action='store_true')
    argparser.set_defaults(use_rave=False)
    argparser.add_argument('--alpha', default=1.0, type=float, help='Weight for b/c (b > alpha * c)')
    argparser.add_argument('--beta', default=1.0, type=float, help='Weight for b/l (b > beta * l)')
    return argparser.parse_args()

def F1(arr, target):
    tp = sum(arr[pred][target] for pred in range(3) if target == pred)
    fp = sum(arr[pred][target] for pred in range(3) if pred == target and target != pred)
    fn = sum(arr[target][pred] for pred in range(3) if target != pred)
    return 2 * tp / (2 * tp + fp + fn)

def evaluate(data, model, use_rave=False):
    model.eval()
    all_jet_predictions = []

    for batch in data:
        sets, jet_features = (batch[0], batch[3])
        sets = sets.to(DEVICE, torch.float)
        if use_rave:
            rave_input = batch[5]
            jet_prediction = model(jet_features, sets, rave_input).cpu().data.numpy()
        else:
            jet_prediction = model(jet_features, sets).cpu().data.numpy()
        all_jet_predictions.append(jet_prediction)

    return np.concatenate(all_jet_predictions)

def main():
    config = parse_args()
    start_time = datetime.now()

    # Load test data
    print('Loading test data...', end='', flush=True)
    test_data = jets_loader.JetGraphDataset('test', debug_load=False, add_jet_flav=True, add_rave_file=config.use_rave)

    vertexing_config = {
        'in_features': 10,
        'out_features': 1,
        'set_fn_feats': [256, 256, 256, 256, 5],
        'method': 'lin5',
        'hidden_mlp': [256],
        'predict_diagonal': False,
        'attention': True,
        'set_model_type': 'deepset'
    } if config.vertexing_model_type == 's2g' else {}

    model = JetClassifier(10, vertexing_config, vertexing_type=config.vertexing_model_type)
    model.load_state_dict(torch.load(config.path_to_trained_model, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    predictions = []
    indx_list = []
    max_batch_size = config.bs

    # Process data in batches
    for tracks_in_jet in tqdm(range(2, np.amax(test_data.n_nodes) + 1)):
        trk_indxs = np.where(np.array(test_data.n_nodes) == tracks_in_jet)[0]
        if len(trk_indxs) < 1:
            continue
        indx_list.extend(trk_indxs)

        sub_batches = np.array_split(trk_indxs, len(trk_indxs) // max_batch_size + 1)
        for sub_batch in sub_batches:
            sets, jet_features = zip(*(test_data[i][:4:3] for i in sub_batch))
            sets, jet_features = torch.stack(list(map(torch.tensor, sets))), torch.stack(list(map(torch.tensor, jet_features)))
            with torch.no_grad():
                if config.use_rave:
                    rave_inputs = torch.stack([test_data[i][5] for i in sub_batch])
                    jet_predictions = model(jet_features, sets, rave_inputs)
                else:
                    jet_predictions = model(jet_features, sets)
            predictions.extend(jet_predictions.cpu().data.numpy())

    # Sorting predictions
    sorted_predictions = [x for _, x in sorted(zip(indx_list, predictions))]
    softmax_predictions = F.softmax(torch.tensor(sorted_predictions), dim=1) * torch.tensor([1., config.alpha, config.beta])
    sorted_predictions = softmax_predictions

    np_matrix = np.zeros((3, 3))
    jet_label = test_data.jet_flavs
    pred = torch.argmax(torch.tensor(sorted_predictions), dim=1)

    accs = np.zeros(4)
    for flav in range(3):
        correct = (pred[jet_label == flav] == jet_label[jet_label == flav]).sum().item()
        accs[flav] = correct / len(jet_label[jet_label == flav])
        for i in range(len(pred)):
            np_matrix[pred[i], jet_label[i]] += 1

    accs[3] = (pred == jet_label).sum().item() / len(jet_label)

    # Print results
    print(np_matrix.astype(int).tolist())
    rate_label_arr = (np_matrix / np_matrix.sum(axis=0)).T
    print(rate_label_arr.diagonal().tolist() + [rate_label_arr.diagonal().mean()])
    rate_pred_arr = np_matrix / np_matrix.sum(axis=1)[:, None]
    print(rate_pred_arr.diagonal().tolist() + [rate_pred_arr.diagonal().mean()])
    f1_scores = [F1(np_matrix, i) for i in range(3)]
    print(f1_scores + [np.mean(f1_scores)])

    # Histogram generation
    pt_ft_list = (75.95093, 49.134453)
    jet_pt_arr = [test_data[i][-2][0] * pt_ft_list[1] + pt_ft_list[0] for i in range(test_data.n_jets)]
    pt_bins = [10., 15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 85., 100.]
    with uproot.recreate("eval_jc.root") as file:
        labels, preds = jet_label == 0, pred == 0
        file['hist_pt_label-b'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=labels.astype(float))
        file['hist_pt_pred-b'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=preds.astype(float))
        file['hist_pt_correct-b'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=(labels & preds).astype(float))
        labels, preds = jet_label == 1, pred == 1
        file['hist_pt_label-c'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=labels.astype(float))
        file['hist_pt_pred-c'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=preds.astype(float))
        file['hist_pt_correct-c'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=(labels & preds).astype(float))
        labels, preds = jet_label == 2, pred == 2
        file['hist_pt_label-l'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=labels.astype(float))
        file['hist_pt_pred-l'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=preds.astype(float))
        file['hist_pt_correct-l'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=(labels & preds).astype(float))
        label_pred_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        for label, pred in label_pred_pairs:
            file[f'hist_pt_label-{label}-pred-{pred}'] = np.histogram(jet_pt_arr, bins=pt_bins, weights=(jet_label == label & pred == pred).astype(float))
        file['tree'] = uproot.newtree({'jet_flav': 'int', 'jet_pt': 'float32', 'softmax-b': 'float32', 'softmax-c': 'float32', 'softmax-l': 'float32'})
        file['tree'].extend({'jet_flav': jet_label, 'jet_pt': jet_pt_arr, 'softmax-b': softmax_predictions[:, 0], 'softmax-c': softmax_predictions[:, 1], 'softmax-l': softmax_predictions[:, 2]})

if __name__ == '__main__':
    main()
