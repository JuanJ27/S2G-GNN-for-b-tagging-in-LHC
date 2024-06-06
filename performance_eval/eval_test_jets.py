import torch
import uproot3 as uproot
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import metrics
import mpmath
from dataloaders.jets_loader import JetGraphDataset

def _get_rand_index(labels, predictions):
    """
    Calculate the Rand Index (RI) between the true labels and predictions.
    """
    n_items = len(labels)
    if n_items < 2:
        return 1
    n_pairs = (n_items * (n_items - 1)) / 2
    correct_pairs = 0
    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true) or ((not label_true) and (not pred_true)):
                correct_pairs += 1
    return correct_pairs / n_pairs

def _error_count(labels, predictions):
    """
    Count true positives, false positives, and false negatives.
    """
    n_items = len(labels)
    true_positives = 0
    false_positive = 0
    false_negative = 0

    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if label_true and pred_true:
                true_positives += 1
            if not label_true and pred_true:
                false_positive += 1
            if label_true and not pred_true:
                false_negative += 1
    return true_positives, false_positive, false_negative

def _get_recall(labels, predictions):
    """
    Calculate the recall score.
    """
    true_positives, _, false_negative = _error_count(labels, predictions)
    if true_positives + false_negative == 0:
        return 0
    return true_positives / (true_positives + false_negative)

def _get_precision(labels, predictions):
    """
    Calculate the precision score.
    """
    true_positives, false_positive, _ = _error_count(labels, predictions)
    if true_positives + false_positive == 0:
        return 0
    return true_positives / (true_positives + false_positive)

def _f_measure(labels, predictions):
    """
    Calculate the F1 score.
    """
    precision = _get_precision(labels, predictions)
    recall = _get_recall(labels, predictions)
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall) / (recall + precision)

def eval_jets_on_test_set(model):
    """
    Evaluate the model on the test set and calculate various scores.
    """
    pred = _predict_on_test_set(model)
    test_ds = uproot.open('data/test/test_data.root')
    jet_df = test_ds['tree'].pandas.df(['jet_flav', 'trk_vtx_index'], flatten=False)
    jet_flav = jet_df['jet_flav']
    target = [x for x in jet_df['trk_vtx_index'].values]

    print('Calculating scores on test set... ', end='')
    start = datetime.now()
    model_scores = {}

    # Calculate metrics and store each in a NumPy array with dtype=object
    model_scores['RI'] = np.array([_get_rand_index(t, p) for t, p in zip(target, pred)], dtype=object)
    model_scores['ARI'] = np.array([adjustedRI_onesided(t, p) for t, p in zip(target, pred)], dtype=object)
    model_scores['P'] = np.array([_get_precision(t, p) for t, p in zip(target, pred)], dtype=object)
    model_scores['R'] = np.array([_get_recall(t, p) for t, p in zip(target, pred)], dtype=object)
    model_scores['F1'] = np.array([_f_measure(t, p) for t, p in zip(target, pred)], dtype=object)

    end = datetime.now()
    print(f': {str(end - start).split(".")[0]}')

    flavours = {5: 'b jets', 4: 'c jets', 0: 'light jets'}
    metrics_to_table = ['P', 'R', 'F1', 'RI', 'ARI']
    df = pd.DataFrame(index=flavours.values(), columns=metrics_to_table)

    for flav_n, flav in flavours.items():
        for metric in metrics_to_table:
            mean_metric = np.mean(model_scores[metric][jet_flav == flav_n])
            df.at[flav, metric] = mean_metric

    return df

def _predict_on_test_set(model):
    """
    Generate predictions on the test set using the model.
    """
    test_ds = JetGraphDataset('test')
    model.eval()
    n_tracks = [test_ds[i][0].shape[0] for i in range(len(test_ds))]
    indx_list = []
    predictions = []

    for tracks_in_jet in range(2, np.amax(n_tracks) + 1):
        trk_indxs = np.where(np.array(n_tracks) == tracks_in_jet)[0]
        if len(trk_indxs) < 1:
            continue
        indx_list += list(trk_indxs)
        input_batch = torch.stack([torch.tensor(test_ds[i][0]) for i in trk_indxs])

        edge_vals = model(input_batch).squeeze(1)
        edge_scores = 0.5 * (edge_vals + edge_vals.transpose(1, 2))
        edge_scores = torch.sigmoid(edge_scores)
        edge_scores[:, np.arange(tracks_in_jet), np.arange(tracks_in_jet)] = 1

        pred_matrices = compute_clusters_with_partition_score(edge_scores)
        pred_clusters = compute_vertex_assignment(pred_matrices)

        predictions += list(pred_clusters.cpu().data.numpy())

    sorted_predictions = [list(x) for _, x in sorted(zip(indx_list, predictions))]
    return sorted_predictions

def compute_partition_score(edge_scores, pred_matrix):
    """
    Compute partition score based on edge scores and predicted matrix.
    """
    score = -(pred_matrix * torch.log(edge_scores + 1e-10) + (1 - pred_matrix) * torch.log(1 - edge_scores + 1e-10))
    return score.sum(dim=1).sum(dim=1)

def fill_gaps(edge_vals):
    """
    Ensure each node is connected to itself and propagate connections.
    """
    b, n, _ = edge_vals.shape
    with torch.no_grad():
        pred_matrices = edge_vals.ge(0.5).float()
        pred_matrices[:, np.arange(n), np.arange(n)] = 1
        ones_now = pred_matrices.sum()
        ones_before = ones_now - 1
        while ones_now != ones_before:
            ones_before = ones_now
            pred_matrices = torch.matmul(pred_matrices, pred_matrices)
            pred_matrices = pred_matrices.bool().float()
            ones_now = pred_matrices.sum()
    return pred_matrices

def compute_vertex_assignment(pred_matrix):
    """
    Compute vertex assignment from the predicted matrix.
    """
    b, n, _ = pred_matrix.shape
    pred_matrix[:, np.arange(n), np.arange(n)] = 1
    clusters = -1 * torch.ones((b, n))
    tensor_1 = torch.tensor(1.)
    for i in range(n):
        clusters = torch.where(pred_matrix[:, i] == 1, i * tensor_1, clusters)
    return clusters.long()

def compute_clusters_with_partition_score(edge_scores):
    """
    Compute clusters with partition score based on edge scores.
    """
    B, N, _ = edge_scores.shape
    Ne = int(N * (N - 1) / 2)
    r, c = np.triu_indices(N, 1)
    r = np.tile(r, B)
    c = np.tile(c, B)
    z = np.repeat(np.arange(B), Ne)
    flat_edge_scores = edge_scores[z, r, c].view(B, Ne)

    sorted_values, indices = torch.sort(flat_edge_scores, descending=True)
    final_edge_decision = torch.zeros(B, N, N)
    final_edge_decision[:, np.arange(N), np.arange(N)] = 1
    flat_sorted_edge_decisions = torch.zeros(B, Ne)
    partition_scores = compute_partition_score(edge_scores, final_edge_decision)

    for edge_i in range(Ne):
        temp_edge_decision = flat_sorted_edge_decisions.clone()
        temp_edge_decision[:, edge_i] = torch.where(sorted_values[:, edge_i] > 0.5, torch.tensor(1), torch.tensor(0))
        temp_edge_decision_unsorted = temp_edge_decision.gather(1, indices.argsort(1))
        temp_partition = torch.zeros(B, N, N)
        temp_partition[z, r, c] = temp_edge_decision_unsorted.flatten()
        temp_partition.transpose(2, 1)[z, r, c] = temp_edge_decision_unsorted.flatten()
        temp_partition = fill_gaps(temp_partition)
        temp_partition_scores = compute_partition_score(edge_scores, temp_partition)
        temp_edge_decision[:, edge_i] = torch.where((temp_partition_scores < partition_scores) & (sorted_values[:, edge_i] > 0.5), torch.tensor(1), torch.tensor(0))
        flat_sorted_edge_decisions = temp_edge_decision
        temp_edge_decision_unsorted = temp_edge_decision.gather(1, indices.argsort(1))
        final_edge_decision[z, r, c] = temp_edge_decision_unsorted.flatten()
        final_edge_decision.transpose(2, 1)[z, r, c] = temp_edge_decision_unsorted.flatten()
        partition_scores = compute_partition_score(edge_scores, fill_gaps(final_edge_decision))

    final_edge_decision = fill_gaps(final_edge_decision)
    return final_edge_decision

def infer_clusters(edge_vals):
    """
    Infer the clusters by enforcing symmetry.
    """
    b, n, _ = edge_vals.shape
    with torch.no_grad():
        pred_matrices = edge_vals + edge_vals.transpose(1, 2)
        pred_matrices = pred_matrices.ge(0.).float()
        pred_matrices[:, np.arange(n), np.arange(n)] = 1
        ones_now = pred_matrices.sum()
        ones_before = ones_now - 1
        while ones_now != ones_before:
            ones_before = ones_now
            pred_matrices = torch.matmul(pred_matrices, pred_matrices)
            pred_matrices = pred_matrices.bool().float()
            ones_now = pred_matrices.sum()

        clusters = -1 * torch.ones((b, n), device=edge_vals.device)
        tensor_1 = torch.tensor(1., device=edge_vals.device)
        for i in range(n):
            clusters = torch.where(pred_matrices[:, i] == 1, i * tensor_1, clusters)
    return clusters.long()

def rand_index(labels, predictions):
    """
    Calculate the Rand Index for clustering performance.
    """
    n_items = len(labels)
    if n_items < 2:
        return 1
    n_pairs = (n_items * (n_items - 1)) / 2
    correct_pairs = 0
    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true) or ((not label_true) and (not pred_true)):
                correct_pairs += 1
    return correct_pairs / n_pairs

def Expval(labels):
    """
    Calculate the expected value for Adjusted Rand Index.
    """
    labels = np.array(labels)
    n = len(labels)
    nchoose2 = (n * (n - 1)) / 2.0
    k = len(list(set(labels)))
    g = [len(np.where(labels == x)[0]) for x in list(set(labels))]
    bn = float(mpmath.bell(n))
    bnmin1 = float(mpmath.bell(n - 1))
    bnratio = bnmin1 / bn
    q = np.sum([(gi * (gi - 1)) / 2 for gi in g])
    return bnratio * (q / nchoose2) + (1 - bnratio) * (1 - (q / nchoose2))

def adjustedRI_onesided(labels, predictions):
    """
    Calculate the Adjusted Rand Index (ARI) for clustering performance.
    """
    ri = rand_index(labels, predictions)
    expval = Expval(labels)
    return (ri - expval) / (1 - expval)

def Error_count(labels, predictions):
    """
    Count true positives, false positives, and false negatives.
    """
    n_items = len(labels)
    true_positives = 0
    false_positive = 0
    false_negative = 0

    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if label_true and pred_true:
                true_positives += 1
            if not label_true and pred_true:
                false_positive += 1
            if label_true and not pred_true:
                false_negative += 1
    return true_positives, false_positive, false_negative

def Recall(labels, predictions):
    """
    Calculate the recall score.
    """
    true_positives, _, false_negative = Error_count(labels, predictions)
    if true_positives + false_negative == 0:
        return 0
    return true_positives / (true_positives + false_negative)

def Precision(labels, predictions):
    """
    Calculate the precision score.
    """
    true_positives, false_positive, _ = Error_count(labels, predictions)
    if true_positives + false_positive == 0:
        return 0
    return true_positives / (true_positives + false_positive)

def f_measure(labels, predictions):
    """
    Calculate the F1 score.
    """
    precision = Precision(labels, predictions)
    recall = Recall(labels, predictions)
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall) / (recall + precision)
