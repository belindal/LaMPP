from typing import Union
import numpy as np
import os
from torch import nn
import torch
from tqdm import tqdm


image_w = 640
image_h = 480

rednet_rooms = ["bathroom", "bedroom", "living room",
                "office", "kitchen"]

med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_colours = [(0, 0, 0),
                 # 0=background
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]

sunrgb_items = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk',
    'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
    'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel',
    'shower_curtain', 'box', 'whiteboard', 'person', 'night_stand',
    'toilet', 'sink', 'lamp', 'bathtub', 'bag',
]


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


class ConfusionMatrix(object):
    """Calculates confusion matrix for multi-class data.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.
    Args:
        num_classes (int): number of classes. See notes for more details.
        average (str, optional): confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
    Note:
        In case of the targets `y` in `(batch_size, ...)` format, target indices between 0 and `num_classes` only
        contribute to the confusion matrix and others are neglected. For example, if `num_classes=20` and target index
        equal 255 is encountered, then it is filtered out.
    """

    def __init__(self, num_classes, average=None):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of ['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        self.confusion_matrix_probs = None

    def reset(self):
        # [actual_label][pred_label]
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes,
                                            dtype=torch.int64, device='cpu')
        # [actual_label][pred_label]
        self.confusion_matrix_probs = torch.zeros(self.num_classes, self.num_classes,
                                                  dtype=torch.double, device='cpu')
        self._num_examples = 0

    def _check_shape(self, output):
        y_pred, y = output

        if y_pred.ndimension() < 2:
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...), "
                             "but given {}".format(y_pred.shape))

        if y_pred.shape[1] != self.num_classes:
            raise ValueError("y_pred does not have correct number of categories: {} vs {}"
                             .format(y_pred.shape[1], self.num_classes))

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...) and y must have "
                             "shape of (batch_size, ...), "
                             "but given {} vs {}.".format(y.shape, y_pred.shape))

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    def update(self, output):
        self._check_shape(output)
        y_pred_probs, y = output

        self._num_examples += y_pred_probs.shape[0]

        # target is (batch_size, ...)
        y_pred = torch.argmax(y_pred_probs, dim=1).flatten().to('cpu')
        y = y.flatten().to('cpu')
        y_pred_probs = y_pred_probs.to('cpu').squeeze(0).view(y_pred_probs.shape[1],-1)

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]
        y_pred_probs = y_pred_probs[:,target_mask]
        for y_i in y.unique():
            category_distributions = y_pred_probs[:, y == y_i].sum(-1)
            self.confusion_matrix_probs[y_i] += category_distributions
            # m = torch.bincount(y_pred[y == y_i], minlength=self.num_classes)
            # self.confusion_matrix[y_i] += m

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    def _get_double_cm(self):
        cm = self.confusion_matrix.type(torch.DoubleTensor)
        if self.average:
            if self.average == "samples":
                return cm / self._num_examples
            elif self.average == "recall":
                return cm / (cm.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return cm / (cm.sum(dim=0) + 1e-15)

        return cm

    def avg_prob(self, ignore_index=None):
        cm = self._get_double_cm()
        cm_probs = self.confusion_matrix_probs.type(torch.DoubleTensor)
        # actual_label, pred_label
        avg_probs = cm_probs / (cm.sum(dim=1).unsqueeze(-1) + 1e-15)
        return avg_probs.diag()

    def iou(self, ignore_index=None):
        cm = self._get_double_cm()

        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)

        if ignore_index is not None:
            indices = list(range(len(iou)))
            indices.remove(ignore_index)
            return iou[indices]

        return iou

    def miou(self, ignore_index=None):
        return self.iou(ignore_index=ignore_index).mean()

    def accuracy(self):
        cm = self._get_double_cm()
        return cm.diag().sum() / (cm.sum() + 1e-15)


def eval_metrics(pred_probs: Union[torch.Tensor, np.array], gt_seg: Union[torch.Tensor, np.array], unk_idx: int=None):
    if type(pred_probs) == torch.Tensor:
        pred_probs = pred_probs.detach().cpu().numpy()
    pred_probs = pred_probs.astype(float)
    pred_seg = pred_probs.argmax(0)
    if type(gt_seg) == torch.Tensor:
        gt_seg = gt_seg.cpu().numpy()
    gt_seg = gt_seg.astype(int)

    if unk_idx is not None:
        mask = gt_seg != unk_idx
        pred_probs = pred_probs[:,mask]
        pred_seg = pred_seg[mask]
        gt_seg = gt_seg[mask]
        n_pixels = mask.sum()
    else:
        pred_probs = pred_probs.flatten()
        pred_seg = pred_seg.flatten()
        gt_seg = gt_seg.flatten()
        n_pixels = np.prod(pred_probs.shape)

    avg_pixelwise_prob4goldlabel, perlabel_avgprob = eval_avgprob4goldlabel(pred_probs, gt_seg, n_pixels)
    perlabel_prf1, perlabel_overlap_pred_gt = eval_perlabel_prf1(pred_seg, gt_seg)
    perlabel_IoU = {}
    perlabel_IoU_raw_counts = {}
    for label in perlabel_overlap_pred_gt:
        perlabel_IoU[label] = perlabel_overlap_pred_gt[label][0] / perlabel_overlap_pred_gt[label][1]
        perlabel_IoU_raw_counts[label] = [perlabel_overlap_pred_gt[label][0], perlabel_overlap_pred_gt[label][1]]
    return {
        "pixelwise_acc": eval_pixelwise_acc(pred_seg, gt_seg, n_pixels),
        "perlabel_IoU": perlabel_IoU,
        "perlabel_IoU_raw_counts": perlabel_IoU_raw_counts,
        "perlabel_avgprob": perlabel_avgprob,
        "avg_pixelwise_prob4goldlabel": avg_pixelwise_prob4goldlabel,
    }


def eval_pixelwise_acc(pred_seg: np.array, gt_seg: np.array, n_pixels: int=None):
    return (pred_seg == gt_seg).sum() / n_pixels


def eval_perlabel_prf1(pred_seg: np.array, gt_seg: np.array):
    perlabel_prf1 = {}
    perlabel_overlap_pred_gt = {}
    all_pred_labels = set(np.unique(pred_seg))
    all_gt_labels = set(np.unique(gt_seg))
    all_labels = all_pred_labels.union(all_gt_labels)
    for label in all_labels:
        pred_pixels = pred_seg == label
        gt_pixels = gt_seg == label
        overlap = (pred_pixels & gt_pixels).sum()
        union = (pred_pixels | gt_pixels).sum()
        precision = overlap / pred_pixels.sum() if pred_pixels.sum() > 0 else 0
        recall = overlap / gt_pixels.sum() if gt_pixels.sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # if recall != recall:
        #     continue
        perlabel_prf1[label] = [precision, recall, f1]
        perlabel_overlap_pred_gt[label.item()] = [overlap.item(), union.item(), pred_pixels.sum().item(), gt_pixels.sum().item()]
    return perlabel_prf1, perlabel_overlap_pred_gt


def eval_avgprob4goldlabel(pred_probs: np.array, gt_seg: np.array, n_pixels: int=None):
    pixelwise_prob4goldlabel = pred_probs[gt_seg,np.arange(gt_seg.shape[0])]
    avg_pixelwise_prob4goldlabel = pixelwise_prob4goldlabel.sum() / n_pixels
    perlabel_avgprob = {}
    for label in np.unique(gt_seg):
        perlabel_pixelwise_prob = pixelwise_prob4goldlabel[gt_seg == label]
        perlabel_avgprob[label] = perlabel_pixelwise_prob.sum() / perlabel_pixelwise_prob.shape[0]
    return avg_pixelwise_prob4goldlabel, perlabel_avgprob


def color_label(label, label_colours=label_colours):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)

