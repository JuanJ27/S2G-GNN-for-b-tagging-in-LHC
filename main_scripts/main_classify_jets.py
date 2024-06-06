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
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-e', '--epochs', default=400, type=int, help='The number of epochs to run')
    argparser.add_argument('-l', '--lr', default=0.0005, type=float, help='The learning rate')
    argparser.add_argument('-b', '--bs', default=1000, type=int, help='Batch size to use')
    argparser.add_argument('--res_dir', default='experiments/jets_results', help='Results directory')
    argparser.add_argument('--pretrained_vertexing_model', default=None, help='Path to trained model')
    argparser.add_argument('--pretrained_vertexing_model_type', default=None, help='s2g')
    argparser.add_argument('--use_rave', dest='use_rave', action='store_true')
    argparser.add_argument('--debug_load', dest='debug_load', action='store_true', help='Load only a small subset of the data')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False, use_rave=False)
    return argparser.parse_args()

def calc_metrics(jet_prediction, jet_label, accum_info, batch_size):
    with torch.no_grad():
        pred = torch.argmax(jet_prediction, dim=1)
        for flav, flav_name in zip([0, 1, 2], ['b', 'c', 'u']):
            correct = (pred[jet_label == flav] == jet_label[jet_label == flav]).sum().item()
            total = len(jet_label[jet_label == flav])
            accum_info[f'accuracy_{flav_name}'] += correct / total
        accum_info['accuracy'] += (pred == jet_label).sum().item()
    return accum_info

def get_loss(y_hat, y):
    return F.cross_entropy(y_hat, y)

def train(data, model, optimizer, use_rave=False):
    return do_epoch(data, model, optimizer, use_rave)

def evaluate(data, model, use_rave=False):
    return do_epoch(data, model, optimizer=None, use_rave=use_rave)

def do_epoch(data, model, optimizer=None, use_rave=False):
    model.train() if optimizer else model.eval()
    start_time = datetime.now()
    accum_info = {k: 0.0 for k in ['loss', 'accuracy', 'insts', 'accuracy_b', 'accuracy_c', 'accuracy_u']}
    n_batches = 0

    for batch in data:
        sets, _, _, jet_features, jet_label = (batch[:4] + (batch[5],) if use_rave else batch[:4])
        sets, jet_label = sets.to(DEVICE, torch.float), jet_label.to(DEVICE, torch.long)
        batch_size = sets.shape[0]
        accum_info['insts'] += batch_size
        n_batches += 1

        jet_prediction = model(jet_features, sets, external_edge_vals=batch[5] if use_rave else None)
        loss = get_loss(jet_prediction, jet_label)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info = calc_metrics(jet_prediction, jet_label, accum_info, batch_size)
        accum_info['loss'] += loss.item() * batch_size

    num_insts = accum_info.pop('insts')
    accum_info['loss'] /= num_insts
    accum_info['accuracy'] /= num_insts
    for flav_name in ['b', 'c', 'u']:
        accum_info[f'accuracy_{flav_name}'] /= n_batches
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
    print(flush=True)

    # Load data
    print('Loading training data...', end='', flush=True)
    train_data = jets_loader.get_data_loader('train', config.bs, config.debug_load, add_jet_flav=True, add_rave_file=config.use_rave)
    print('Loading validation data...', end='', flush=True)
    val_data = jets_loader.get_data_loader('validation', config.bs, config.debug_load, add_jet_flav=True, add_rave_file=config.use_rave)

    # Create model instance
    vertexing_config = {
        's2g': {'in_features': 10, 'out_features': 1, 'set_fn_feats': [256, 256, 256, 256, 5], 'method': 'lin5', 'hidden_mlp': [256], 'predict_diagonal': False, 'attention': True, 'set_model_type': 'deepset'}
    }.get(config.pretrained_vertexing_model_type, {})

    model = JetClassifier(10, vertexing_config, vertexing_type=config.pretrained_vertexing_model_type)
    if config.pretrained_vertexing_model:
        vertexing_model_state_dict = torch.load(config.pretrained_vertexing_model, map_location='cpu')
        model_state_dict = model.state_dict()
        model_state_dict.update({f'vertexing.{k}': v for k, v in vertexing_model_state_dict.items()})
        model.load_state_dict(model_state_dict)
        for name, p in model.named_parameters():
            if 'vertexing.' in name:
                p.requires_grad = False

    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The number of model parameters is {num_params}')

    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)

    # Metrics
    train_loss, train_acc = np.empty(config.epochs, float), np.empty(config.epochs, float)
    val_loss, val_acc = np.empty(config.epochs, float), np.empty(config.epochs, float)

    best_epoch, best_val_acc, best_model = -1, -1, None

    # Training and evaluation process
    for epoch in range(1, config.epochs + 1):
        train_info = train(train_data, model, optimizer, use_rave=config.use_rave)
        print(f"\tTraining - {epoch:4} loss:{train_info['loss']:.6f} -- mean_acc:{train_info['accuracy']:.4f} -- mean_b_acc:{train_info['accuracy_b']:.4f} -- mean_c_acc:{train_info['accuracy_c']:.4f} -- mean_light_acc:{train_info['accuracy_u']:.4f}", flush=True)
        train_loss[epoch-1], train_acc[epoch-1] = train_info['loss'], train_info['accuracy']

        val_info = evaluate(val_data, model, use_rave=config.use_rave)
        print(f"\tVal      - {epoch:4} loss:{val_info['loss']:.6f} -- mean_acc:{val_info['accuracy']:.4f} -- mean_b_acc:{val_info['accuracy_b']:.4f} -- mean_c_acc:{val_info['accuracy_c']:.4f} -- mean_light_acc:{val_info['accuracy_u']:.4f} -- runtime:{val_info['run_time']}\n", flush=True)
        val_loss[epoch-1], val_acc[epoch-1] = val_info['loss'], val_info['accuracy']

        if val_info['accuracy'] > best_val_acc:
            best_val_acc = val_info['accuracy']
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if best_epoch < epoch - 20:
            print('Early stopping training due to no improvement over the last 20 epochs...')
            break

    del train_data, val_data
    print(f'Best validation acc: {best_val_acc:.4f}, best epoch: {best_epoch}.')
    print(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')
    print()

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
        results_dict = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}
        pd.DataFrame(results_dict).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        pd.DataFrame({'best_val_acc': [best_val_acc], 'best_epoch': [best_epoch]}).to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)
        with open(os.path.join(output_dir, 'used_config.json'), 'w') as fp:
            json.dump(vars(config), fp)

    print(f'Total runtime: {str(datetime.now() - start_time).split(".")[0]}')

if __name__ == '__main__':
    main()
