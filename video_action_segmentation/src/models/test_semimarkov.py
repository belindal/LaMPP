import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from torch_struct import SemiMarkov, MaxSemiring

from models.semimarkov.semimarkov_modules import SemiMarkovModule

# device = torch.device("cuda")
device = torch.device("cpu")

sm_max = SemiMarkov(MaxSemiring)

BIG_NEG = -1e9


class ToyDataset(Dataset):
    def __init__(self, labels, features, lengths, valid_classes, max_k):
        self.labels = labels
        self.features = features
        self.lengths = lengths
        self.valid_classes = valid_classes
        self.max_k = max_k

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        labels = self.labels[index]
        spans = SemiMarkovModule.labels_to_spans(labels.unsqueeze(0), max_k=self.max_k).squeeze(0)
        return {
            'labels': self.labels[index],
            'features': self.features[index],
            'lengths': self.lengths[index],
            'valid_classes': self.valid_classes[index],
            'spans': spans,
        }


def synthetic_data(num_data_points=200, C=3, N=100, K=5, num_classes_per_instance=None):
    def make_synthetic_features(class_labels, shift_constant=1.0):
        _batch_size, _N = class_labels.size()
        f = torch.randn((_batch_size, _N, C))
        shift = torch.zeros_like(f)
        shift.scatter_(2, class_labels.unsqueeze(2), shift_constant)
        return shift + f

    labels_l = []
    lengths = []
    valid_classes = []
    for i in range(num_data_points):
        if i == 0:
            length = N
        else:
            length = random.randint(K, N)
        lengths.append(length)
        lab = []
        current_step = 0
        if num_classes_per_instance is not None:
            assert num_classes_per_instance <= C
            this_valid_classes = np.random.choice(list(range(C)), size=num_classes_per_instance, replace=False)
        else:
            this_valid_classes = list(range(C))
        valid_classes.append(this_valid_classes)
        while len(lab) < N:
            step_length = random.randint(1, K - 1)
            this_label = this_valid_classes[current_step % len(this_valid_classes)]
            lab.extend([this_label] * step_length)
            current_step += 1
        lab = lab[:N]
        labels_l.append(lab)
    labels = torch.LongTensor(labels_l)
    features = make_synthetic_features(labels)
    lengths = torch.LongTensor(lengths)
    valid_classes = [torch.LongTensor(tvc) for tvc in valid_classes]

    return labels, features, lengths, valid_classes


def partition_rows(arr, N):
    if isinstance(arr, list):
        assert N < len(list)
    else:
        assert N < arr.size(0)
    return arr[:N], arr[N:]


def test_learn_synthetic():
    C = 3
    MAX_K = 20
    K = 5
    N = 20
    N_train = 150
    N_test = 50

    closed_form_supervised = True

    supervised = True

    allow_self_transitions = True

    num_classes_per_instance = None

    epochs = 20

    batch_size = 10

    train_data = ToyDataset(
        *synthetic_data(num_data_points=N_train, C=C, N=N, K=K, num_classes_per_instance=num_classes_per_instance),
        max_k=MAX_K
    )
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_data = ToyDataset(
        *synthetic_data(num_data_points=N_test, C=C, N=N, K=K, num_classes_per_instance=num_classes_per_instance),
        max_k=MAX_K
    )
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = SemiMarkovModule(C, C, max_k=MAX_K, allow_self_transitions=allow_self_transitions)
    model.initialize_gaussian(train_data.features, train_data.lengths)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    if supervised and closed_form_supervised:
        train_features = []
        train_labels = []
        for i in range(len(train_data)):
            sample = train_data[i]
            train_features.append(sample['features'])
            train_labels.append(sample['labels'])
        model.fit_supervised(train_features, train_labels)
    else:
        for epoch in range(epochs):
            losses = []
            for batch in train_loader:
                # if self.args.cuda:
                #     features = features.cuda()
                #     task_indices = task_indices.cuda()
                #     gt_single = gt_single.cuda()
                features = batch['features']
                lengths = batch['lengths']
                spans = batch['spans']
                valid_classes = batch['valid_classes']
                this_N = lengths.max().item()
                features = features[:, :this_N, :]
                spans = spans[:, :this_N]

                if supervised:
                    spans_sup = spans
                else:
                    spans_sup = None

                this_loss = -model.log_likelihood(features, lengths, valid_classes_per_instance=valid_classes,
                                                  spans=spans_sup)
                this_loss.backward()

                losses.append(this_loss.item())

                optimizer.step()
                model.zero_grad()
            train_acc, train_remap_acc, _ = predict_synthetic(model, train_loader)
            test_acc, test_remap_acc, _ = predict_synthetic(model, test_loader)
            # print(train_acc)
            # print(train_remap_acc)
            # print(test_acc)
            # print(test_remap_acc)
            print("epoch {} avg loss: {:.4f}\ttrain acc: {:.2f}\ttest acc: {:.2f}".format(
                epoch,
                np.mean(losses),
                train_acc if supervised else train_remap_acc,
                test_acc if supervised else test_remap_acc,
            ))

    return model, train_loader, test_loader


def optimal_map(predicted_labels, gold_labels, possible_labels):
    assert all(lab in possible_labels for lab in predicted_labels)
    assert all(lab in possible_labels for lab in gold_labels)
    voting_table = np.zeros((len(possible_labels), len(possible_labels)))
    labs_numpy = possible_labels.detach().cpu().numpy()
    for idx_gt, label_gt in enumerate(labs_numpy):
        gold_mask = gold_labels == label_gt
        for idx_pr, label_pr in enumerate(labs_numpy):
            voting_table[idx_gt, idx_pr] = (predicted_labels[gold_mask] == label_pr).sum()

    best_gt, best_pr = linear_sum_assignment(-voting_table)
    mapping = {
        labs_numpy[pr]: labs_numpy[gt]
        for pr, gt in zip(best_pr, best_gt)
    }
    remapped = predicted_labels.clone()
    remapped.apply_(lambda lab: mapping[lab])
    return remapped, mapping


def predict_synthetic(model, dataloader):
    items = []
    token_match = 0
    token_total = 0
    token_remap_match = 0
    for batch in dataloader:
        features = batch['features']
        lengths = batch['lengths']
        gold_spans = batch['spans']
        valid_classes = batch['valid_classes']

        batch_size = features.size(0)

        this_N = lengths.max().item()
        features = features[:, :this_N, :]
        gold_spans = gold_spans[:, :this_N]

        pred_spans = model.viterbi(features, lengths, valid_classes_per_instance=valid_classes, add_eos=True)
        gold_labels = model.spans_to_labels(gold_spans)
        pred_labels = model.spans_to_labels(pred_spans)

        gold_labels_trim = model.trim(gold_labels, lengths, check_eos=False)
        pred_labels_trim = model.trim(pred_labels, lengths, check_eos=True)

        assert len(gold_labels_trim) == batch_size
        assert len(pred_labels_trim) == batch_size

        for i in range(batch_size):
            this_valid_classes = valid_classes[i]
            pred_remapped, mapping = optimal_map(pred_labels_trim[i], gold_labels_trim[i], this_valid_classes)
            item = {
                'length': lengths[i].item(),
                'gold_spans': gold_spans[i],
                'pred_spans': pred_spans[i],
                'gold_labels': gold_labels[i],
                'pred_labels': pred_labels[i],
                'gold_labels_trim': gold_labels_trim[i],
                'pred_labels_trim': pred_labels_trim[i],
                'pred_labels_remap_trim': pred_remapped,
                'mapping': mapping
            }
            items.append(item)
            token_match += (gold_labels_trim[i] == pred_labels_trim[i]).sum().item()
            token_remap_match += (gold_labels_trim[i] == pred_remapped).sum().item()
            token_total += pred_labels_trim[i].size(0)
    accuracy = 100.0 * token_match / token_total
    remapped_accuracy = 100.0 * token_remap_match / token_total
    return accuracy, remapped_accuracy, items


def test_labels_and_spans():
    position_labels = torch.LongTensor([[0, 1, 1, 2, 2, 2], [0, 1, 2, 3, 3, 4]])
    spans = torch.LongTensor([[0, 1, -1, 2, -1, -1], [0, 1, 2, 3, -1, 4]])
    rle = [[(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 1), (2, 1), (3, 2), (4, 1)]]
    assert (SemiMarkovModule.labels_to_spans(position_labels, max_k=10) == spans).all()
    assert (SemiMarkovModule.spans_to_labels(spans) == position_labels).all()
    assert SemiMarkovModule.rle_spans(spans, lengths=torch.LongTensor([6, 6])) == rle
    trunc_lengths = torch.LongTensor([5, 6])
    trunc_rle = [[(0, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 1), (3, 2), (4, 1)]]
    assert SemiMarkovModule.rle_spans(spans, lengths=trunc_lengths) == trunc_rle

    rand_labels = torch.randint(low=0, high=3, size=(5, 20))
    assert (SemiMarkovModule.spans_to_labels(
        SemiMarkovModule.labels_to_spans(rand_labels, max_k=5)) == rand_labels).all()


def test_log_hsmm():
    # b = 100
    # C = 7
    # N = 300
    # K = 50
    # step_length = 20

    # b = 10
    # C = 3
    # N = 10
    # K = 20  # K > N
    # step_length = 2

    b = 10
    C = 4
    N = 100
    K = 5
    step_length = 4

    add_eos = True

    padded_length = N + step_length * 2

    lengths_unpadded = torch.full((b,), N).long()
    lengths_unpadded[0] = padded_length
    lengths = lengths_unpadded + 1

    num_steps = N // step_length
    assert N % step_length == 0  # since we're fixing lengths, need to end perfectly

    # trans_scores = torch.from_numpy(np.array([[0,1,0],[0,0,1],[1,0,0]]).T).float().log()
    trans_scores = torch.zeros(C, C, device=device)
    init_scores = torch.full((C,), BIG_NEG, device=device)
    init_scores[0] = 0

    emission_scores = torch.full((b, padded_length, C), BIG_NEG, device=device)

    for n in range(padded_length):
        c = (n // step_length) % C
        emission_scores[:, n, c] = 1

    length_scores = torch.full((K, C), BIG_NEG, device=device)
    length_scores[step_length, :] = 0

    scores = SemiMarkovModule.log_hsmm(trans_scores, emission_scores, init_scores, length_scores, lengths_unpadded,
                                       add_eos=add_eos)
    marginals = sm_max.marginals(scores, lengths=lengths)

    sequence, extra = sm_max.from_parts(marginals)

    for step in range(num_steps):
        c = step % C
        assert torch.allclose(sequence[:, step_length * step], torch.full((1,), c).long())

    # C == EOS
    if add_eos:
        batch_indices = torch.arange(0, b)
        assert torch.allclose(sequence[batch_indices, lengths - 1], torch.full((1,), C).long())


test_labels_and_spans()
print("test_labels_and_spans passed")

test_log_hsmm()
print("test_log_hsmm passed")

model, trainloader, testloader = test_learn_synthetic()
train_acc, train_remap_accuracy, train_preds = predict_synthetic(model, trainloader)
test_acc, test_remap_accuracy, test_preds = predict_synthetic(model, testloader)
print("train acc: {:.2f}".format(train_acc))
print("train remap acc: {:.2f}".format(train_remap_accuracy))
print("test acc: {:.2f}".format(test_acc))
print("test remap acc: {:.2f}".format(test_remap_accuracy))
