import random
import os
import sys
import argparse
import copy
import shutil
import json
from pprint import pprint
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

"""
How To:
Example for running from command line:
python <path_to>/SetToGraph/main_scripts/main_jets.py
"""
# Change working directory to project's main directory, and add it to path for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from models.set_to_graph import SetToGraph
from dataloaders import jets_loader
from performance_eval.eval_test_jets import eval_jets_on_test_set

DEVICE = 'cpu'  # Not using CUDA

def parse_args():
    """Define and retrieve command line arguments"""
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-e', '--epochs', default=400, type=int, help='The number of epochs to run')
    argparser.add_argument('-l', '--lr', default=0.001, type=float, help='The learning rate')
    argparser.add_argument('-b', '--bs', default=2048, type=int, help='Batch size to use')
    argparser.add_argument('--method', default='lin2', help='Method to transfer from sets to graphs: lin2 for S2G, lin5 for S2G+')
    argparser.add_argument('--res_dir', default='experiments/jets_results', help='Results directory')
    argparser.add_argument('--debug_load', dest='debug_load', action='store_true', help='Load only a small subset of the data')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False)
    return argparser.parse_args()

def calc_metrics(pred_partitions, partitions_as_graph, partitions, accum_info):
    with torch.no_grad():
        B, N = partitions.shape
        C = pred_partitions.max().item() + 1
        pred_onehot = torch.zeros((B, N, C), dtype=torch.float, device=partitions.device)
        pred_onehot.scatter_(2, pred_partitions[:, :, None], 1)
        pred_matrices = torch.matmul(pred_onehot, pred_onehot.transpose(1, 2))

        # Calculate fscore, precision, recall
        tp = (pred_matrices * partitions_as_graph).sum(dim=(1, 2)) - N  # Exclude diagonals
        fp = (pred_matrices * (1 - partitions_as_graph)).sum(dim=(1, 2))
        fn = ((1 - pred_matrices) * partitions_as_graph).sum(dim=(1, 2))
        accum_info['recall'] += (tp / (tp + fp + 1e-10)).sum().item()
        accum_info['precision'] += (tp / (tp + fn + 1e-10)).sum().item()
        accum_info['fscore'] += ((2 * tp) / (2 * tp + fp + fn + 1e-10)).sum().item()

        # Calculate RI
        equiv_pairs = (pred_matrices == partitions_as_graph).float()
        accum_info['accuracy'] += equiv_pairs.mean(dim=(1, 2)).sum().item()
        equiv_pairs[:, torch.arange(N), torch.arange(N)] = 0  # Exclude diagonal pairs
        accum_info['ri'] += equiv_pairs.sum(dim=(1, 2)).item() / (N * (N - 1))
    return accum_info

def infer_clusters(edge_vals):
    """Infer the clusters. Enforce symmetry."""
    b, n, _ = edge_vals.shape
    with torch.no_grad():
        pred_matrices = (edge_vals + edge_vals.transpose(1, 2)).ge(0.).float()
        pred_matrices[:, np.arange(n), np.arange(n)] = 1.
        ones_now = pred_matrices.sum()
        while True:
            pred_matrices = torch.matmul(pred_matrices, pred_matrices).bool().float()
            ones_new = pred_matrices.sum()
            if ones_new == ones_now:
                break
            ones_now = ones_new

        clusters = -1 * torch.ones((b, n), device=edge_vals.device)
        for i in range(n):
            clusters = torch.where(pred_matrices[:, i] == 1, i * torch.ones(1, device=edge_vals.device), clusters)
    return clusters.long()

def get_loss(y_hat, y):
    """Calculate loss excluding diagonal elements"""
    B, N, _ = y_hat.shape
    y_hat[:, torch.arange(N), torch.arange(N)] = torch.finfo(y_hat.dtype).max  # Mask diagonal elements
    loss = F.binary_cross_entropy_with_logits(y_hat, y)
    y_hat = torch.sigmoid(y_hat)
    tp = (y_hat * y).sum(dim=(1, 2))
    fn = ((1 - y_hat) * y).sum(dim=(1, 2))
    fp = (y_hat * (1 - y)).sum(dim=(1, 2))
    return loss - ((2 * tp) / (2 * tp + fp + fn + 1e-10)).sum()

def do_epoch(data, model, optimizer=None):
    model.train() if optimizer else model.eval()
    start_time = datetime.now()
    accum_info = {k: 0.0 for k in ['ri', 'loss', 'accuracy', 'fscore', 'precision', 'recall']}
    for sets, partitions, partitions_as_graph in data:
        sets, partitions, partitions_as_graph = sets.to(DEVICE), partitions.to(DEVICE), partitions_as_graph.to(DEVICE)
        batch_size = sets.shape[0]

        edge_vals = model(sets).squeeze(1)
        pred_partitions = infer_clusters(edge_vals)
        loss = get_loss(edge_vals, partitions_as_graph)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info = calc_metrics(pred_partitions, partitions_as_graph, partitions, accum_info)
        accum_info['loss'] += loss.item() * batch_size
        accum_info['insts'] += batch_size

    num_insts = accum_info.pop('insts')
    for key in accum_info:
        accum_info[key] /= num_insts
    accum_info['run_time'] = str(datetime.now() - start_time).split(".")[0]
    return accum_info

def main():
    start_time = datetime.now()
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    config = parse_args()
    pprint(vars(config))

    # Load data
    print('Loading training data...', end='', flush=True)
    train_data = jets_loader.get_data_loader('train', config.bs, config.debug_load)
    print('Loading validation data...', end='', flush=True)
    val_data = jets_loader.get_data_loader('validation', config.bs, config.debug_load)

    # Create model instance
    model = SetToGraph(10,
                       out_features=1,
                       set_fn_feats=[256, 256, 256, 256, 5],
                       method=config.method,
                       hidden_mlp=[256],
                       predict_diagonal=False,
                       attention=True,
                       set_model_type='deepset')
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The number of model parameters is {num_params}')

    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)

    # Metrics
    train_loss, train_ri = np.empty(config.epochs), np.empty(config.epochs)
    val_loss, val_ri = np.empty(config.epochs), np.empty(config.epochs)

    best_epoch, best_val_fscore, best_model = -1, -1, None

    # Training and evaluation process
    for epoch in range(1, config.epochs + 1):
        train_info = do_epoch(train_data, model, optimizer)
        print(f"\tTraining - {epoch:4} loss:{train_info['loss']:.6f} -- mean_ri:{train_info['ri']:.4f} -- fscore:{train_info['fscore']:.4f} -- recall:{train_info['recall']:.4f} -- precision:{train_info['precision']:.4f} -- runtime:{train_info['run_time']}", flush=True)
        train_loss[epoch-1], train_ri[epoch-1] = train_info['loss'], train_info['ri']

        val_info = do_epoch(val_data, model)
        print(f"\tVal      - {epoch:4} loss:{val_info['loss']:.6f} -- mean_ri:{val_info['ri']:.4f} -- fscore:{val_info['fscore']:.4f} -- recall:{val_info['recall']:.4f} -- precision:{val_info['precision']:.4f}  -- runtime:{val_info['run_time']}\n", flush=True)
        val_loss[epoch-1], val_ri[epoch-1] = val_info['loss'], val_info['ri']

        if val_info['fscore'] > best_val_fscore:
            best_val_fscore = val_info['fscore']
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if best_epoch < epoch - 20:
            print('Early stopping training due to no improvement over the last 20 epochs...')
            break

    del train_data, val_data
    print(f'Best validation F-score: {best_val_fscore:.4f}, best epoch: {best_epoch}.')
    print(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')

    # Saving to disk
    if config.save:
        os.makedirs(config.res_dir, exist_ok=True)
        exp_dir = f'jets_{start_time:%Y%m%d_%H%M%S}_0'
        output_dir = os.path.join(config.res_dir, exp_dir)

        i = 0
        while os.path.exists(output_dir):
            i += 1
            output_dir = f"{output_dir[:-1]}{i}"
            if i > 9:
                print(f'Cannot save results on disk. (tried to save as {output_dir})')
                return

        os.makedirs(output_dir)
        print(f'Saving all to {output_dir}')
        torch.save(best_model.state_dict(), os.path.join(output_dir, "exp_model.pt"))
        shutil.copyfile(__file__, os.path.join(output_dir, 'code.py'))
        results_dict = {'train_loss': train_loss, 'train_ri': train_ri, 'val_loss': val_loss, 'val_ri': val_ri}
        pd.DataFrame(results_dict).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_ri': best_val_fscore, 'best_epoch': best_epoch}
        pd.DataFrame(best_dict, index=[0]).to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)
        with open(os.path.join(output_dir, 'used_config.json'), 'w') as fp:
            json.dump(vars(config), fp)

    # Evaluate on test set
    print(f'Epoch {best_epoch} - evaluating over test set.')
    test_results = eval_jets_on_test_set(best_model)
    print('Test results:')
    print(test_results)
    if config.save:
        test_results.to_csv(os.path.join(output_dir, "test_results.csv"), index=True)

    print(f'Total runtime: {str(datetime.now() - start_time).split(".")[0]}')

if __name__ == '__main__':
    main()

