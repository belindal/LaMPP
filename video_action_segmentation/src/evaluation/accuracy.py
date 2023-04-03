# modified from slim_mallow by Anna Kukleva, https://github.com/Annusha/slim_mallow

import pprint
from collections import defaultdict, Counter
import editdistance


import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.logger import logger


def singleton_lookup(dictionary, label):
    assert label in dictionary, "{} not in {}".format(label, dictionary)
    # this should be a singleton unless 'max' was used for optimization
    values = dictionary[label]
    assert len(values) == 1
    return next(iter(values))

def run_length_encode(labels):
    rle = []
    current_label = None
    count = 0
    for label in labels:
        if current_label is None or label != current_label:
            if current_label is not None:
                assert count > 0
                rle.append((current_label, count))
            count = 0
            current_label = label
        count += 1
    if current_label is not None:
        assert count > 0
        rle.append((current_label, count))
    assert sum(count for sym, count in rle) == len(labels)
    return rle

class Accuracy(object):
    """ Implementation of evaluation metrics for unsupervised learning.

    Since it's unsupervised learning relations between ground truth labels
    and output segmentation should be found.
    Hence the Hungarian method was used and labeling which gives us
    the best score is used as a result.
    """
    def __init__(self, n_frames=1, verbose=True, corpus=None):
        """
        Args:
            n_frames: frequency of sampling,
                in case of it's equal to 1 => dense sampling
        """
        self._n_frames = n_frames
        self._reset()

        self._corpus = corpus


        self._predicted_rle_per_video = [] # : List[List[Tuple[label, length]]], one entry per video
        self._gt_rle_per_video = [] # : List[List[Tuple[label, length]]], one entry per video

        self._predicted_labels_per_video = []
        self._gt_labels_per_video = []
        self._gt_labels_multi_per_video = []

        self._predicted_labels = None
        self._gt_labels_subset = None
        self._gt_labels = None
        self._gt_labels_multi = None
        self._boundaries = None
        # all frames used for alg without any subsampling technique
        self._indices = None

        self._frames_overall = 0

        self._true_background_frames = None
        self._pred_background_frames = None

        self._frames_true_pr = 0
        self._average_score = 0
        self._processed_number = 0
        # self._classes_precision = {}
        self._precision = None
        self._recall = None
        self._precision_without_bg = None
        self._recall_without_bg = None
        self._precision = None
        self._recall = None

        self._multiple_labels = None

        self._classes_recall = {}
        self._classes_MoF = {}
        self._classes_IoU = {}
        self._non_bg_IoU_multi = None
        # keys - gt, values - pr
        self.exclude = {}

        self._classes_levenshtein = {}
        self._classes_step_recall = {}

        self._logger = logger
        self._return = {}

        self._verbose = verbose

    def _reset(self):
        self._n_clusters = 0

        self._gt_label2index = {}
        self._gt_index2label = {}
        self._pr_label2index = {}
        self._pr_index2label = {}

        self._voting_table = []
        self._gt2cluster = defaultdict(list)
        self._acc_per_gt_class = {}

        self.exclude = {}

    def _single_timestep_gt_labels(self, labels):
        # get a single label per timestep
        # should be nested list
        assert isinstance(labels, list) and isinstance(labels[0], list)
        # can have multiple GT labels per timestep, so we need to take just one per timestep
        return [lab_t[0] for lab_t in labels]

    def _add_labels(self, labels, is_predicted: bool):
        if is_predicted:
            rle = run_length_encode(labels)
            self._predicted_labels = None
            self._predicted_labels_per_video.append(labels)
            self._predicted_rle_per_video.append(rle)
        else:
            # ground truth can have multiple labels per timestep; deduplicate
            labels_single = self._single_timestep_gt_labels(labels)
            rle_single = run_length_encode(labels_single)
            self._gt_labels = None
            self._gt_labels_multi = None
            self._gt_labels_subset = None
            self._indices = None
            self._gt_labels_per_video.append(labels_single)
            self._gt_labels_multi_per_video.append(labels)
            self._gt_rle_per_video.append(rle_single)

    def add_gt_labels(self, labels):
        self._add_labels(labels, is_predicted=False)

    def add_predicted_labels(self, labels):
        self._add_labels(labels, is_predicted=True)

    # @property
    # def predicted_labels(self):
    #     return self._predicted_labels
    #
    # @predicted_labels.setter
    # def predicted_labels(self, labels):
    #     self._predicted_labels = np.array(labels)
    #     self._reset()
    #
    # @property
    # def gt_labels(self):
    #     return self._gt_labels_subset
    #
    # @gt_labels.setter
    # def gt_labels(self, labels):
    #     # should be nested list
    #     assert isinstance(labels, list) and isinstance(labels[0], list)
    #     labels = [lab_t[0] for lab_t in labels]
    #     self._gt_labels = np.array(labels)
    #     self._gt_labels_subset = self._gt_labels[:]
    #     self._indices = list(range(len(self._gt_labels)))

    def _set_gt_labels(self):
        labels = [x for xs in self._gt_labels_per_video for x in xs]
        labels_multi = [x for xs in self._gt_labels_multi_per_video for x in xs]
        self._gt_labels = np.array(labels)
        self._gt_labels_subset = self._gt_labels[:]
        self._gt_labels_multi = labels_multi
        assert len(labels) == len(labels_multi)
        self._indices = list(range(len(self._gt_labels)))

    def _set_predicted_labels(self):
        labels = [x for xs in self._predicted_labels_per_video for x in xs]
        self._predicted_labels = np.array(labels)

    @property
    def gt_labels(self):
        if self._gt_labels is None:
            self._set_gt_labels()
        return self._gt_labels

    @property
    def gt_labels_multi(self):
        if self._gt_labels_multi is None:
            self._set_gt_labels()
        return self._gt_labels_multi

    @property
    def gt_labels_subset(self):
        if self._gt_labels_subset is None:
            self._set_gt_labels()
        return self._gt_labels_subset

    @property
    def indices(self):
        if self._indices is None:
            self._set_gt_labels()
        return self._indices

    @property
    def predicted_labels(self):
        if self._predicted_labels is None:
            self._set_predicted_labels()
        return self._predicted_labels

    # @property
    # def params(self):
    #     """
    #     boundaries: if frames samples from segments we need to know boundaries
    #         of these segments to fulfill them after
    #     indices: frames extracted for whatever and indeed evaluation
    #     """
    #     return self._boundaries, self._indices
    #
    # @params.setter
    # def params(self, params):
    #     self._boundaries = params[0]
    #     self._indices = params[1]
    #     self._gt_labels_subset = self._gt_labels[self._indices]

    def _create_voting_table(self):
        """Filling table with assignment scores.

        Create table which represents paired label assignments, i.e. each
        cell comprises score for corresponding label assignment"""
        size = max(len(np.unique(self.gt_labels_subset)),
                   len(np.unique(self.predicted_labels)))
        self._voting_table = np.zeros((size, size))

        for idx_gt, gt_label in enumerate(np.unique(self.gt_labels_subset)):
            self._gt_label2index[gt_label] = idx_gt
            self._gt_index2label[idx_gt] = gt_label

        if len(self._gt_label2index) < size:
            for idx_gt in range(len(np.unique(self.gt_labels_subset)), size):
                gt_label = idx_gt
                while gt_label in self._gt_label2index:
                    gt_label += 1
                self._gt_label2index[gt_label] = idx_gt
                self._gt_index2label[idx_gt] = gt_label

        for idx_pr, pr_label in enumerate(np.unique(self.predicted_labels)):
            self._pr_label2index[pr_label] = idx_pr
            self._pr_index2label[idx_pr] = pr_label

        if len(self._pr_label2index) < size:
            for idx_pr in range(len(np.unique(self.predicted_labels)), size):
                pr_label = idx_pr
                while pr_label in self._pr_label2index:
                    pr_label += 1
                self._pr_label2index[pr_label] = idx_pr
                self._pr_index2label[idx_pr] = pr_label

        for idx_gt, gt_label in enumerate(np.unique(self.gt_labels_subset)):
            if gt_label in list(self.exclude.keys()):
                continue
            gt_mask = self.gt_labels_subset == gt_label
            for idx_pr, pr_label in enumerate(np.unique(self.predicted_labels)):
                if pr_label in list(self.exclude.values()):
                    continue
                self._voting_table[idx_gt, idx_pr] = \
                    np.sum(self.predicted_labels[gt_mask] == pr_label, dtype=float)
        for key, val in self.exclude.items():
            # works only if one pair in exclude
            assert len(self.exclude) == 1
            try:
                self._voting_table[self._gt_label2index[key], self._pr_label2index[val[0]]] = size * np.max(self._voting_table)
            except KeyError:
                logger.debug('No background!')
                self._voting_table[self._gt_label2index[key], -1] = size * np.max(self._voting_table)
                self._pr_index2label[size - 1] = val[0]
                self._pr_label2index[val[0]] = size - 1

    def _create_correspondences(self, method='hungarian', optimization='max'):
        """ Find output labels which correspond to ground truth labels.

        Hungarian method finds one-to-one mapping: if there is squared matrix
        given, then for each output label -> gt label. If not, some labels will
        be without correspondences.
        Args:
            method: hungarian or max
            optimization: for hungarian method usually min problem but here
                is max, hence convert to min
            where: if some actions are not in the video collection anymore
        """
        if method == 'hungarian':
            try:
                assert self._voting_table.shape[0] == self._voting_table.shape[1]
            except AssertionError:
                self._logger.debug('voting table non squared')
                raise AssertionError('bum tss')
            if optimization == 'max':
                # convert max problem to minimization problem
                self._voting_table *= -1
            x, y = linear_sum_assignment(self._voting_table)
            for idx_gt, idx_pr in zip(x, y):
                self._gt2cluster[self._gt_index2label[idx_gt]] = [self._pr_index2label[idx_pr]]
        elif method == 'max':
            # maximum voting, won't create exactly one-to-one mapping
            max_responses = np.argmax(self._voting_table, axis=0)
            for idx, c in enumerate(max_responses):
                # c is index of gt label
                # idx is predicted cluster label
                self._gt2cluster[self._gt_index2label[c]].append(idx)
        elif method == 'identity':
            for label in np.unique(self.gt_labels_subset):
                self._gt2cluster[label] = [label]

    def _fulfill_segments_nondes(self, boundaries, predicted_labels, n_frames):
        full_predicted_labels = []
        for idx, slice in enumerate(range(0, len(predicted_labels), n_frames)):
            start, end = boundaries[idx]
            label_counter = Counter(predicted_labels[slice: slice + n_frames])
            win_label = label_counter.most_common(1)[0][0]
            full_predicted_labels += [win_label] * (end - start + 1)
        return np.asarray(full_predicted_labels)

    def _fulfill_segments(self):
        """If was used frame sampling then anyway we need to get assignment
        for each frame"""
        self._full_predicted_labels = self._fulfill_segments_nondes(self._boundaries, self.predicted_labels, self._n_frames)

    def compute_assignment(self, optimal_assignment: bool, optimization='max', possible_gt_labels=None):
        self._n_clusters = len(np.unique(self.predicted_labels))
        if optimal_assignment:
            self._create_voting_table()
            self._create_correspondences(method='hungarian', optimization=optimization)
        else:
            self._create_correspondences(method='identity')

        if possible_gt_labels is None:
            possible_gt_labels = np.unique(self.gt_labels_subset)

        num_gt_labels = len(possible_gt_labels)
        num_pr_labels = len(np.unique(self.predicted_labels))

        assert num_pr_labels <= num_gt_labels, "gt_labels: {}, pred_labels: {}".format(
            possible_gt_labels,
            np.unique(self.predicted_labels),
        )

        if self._verbose:
            self._logger.debug('# gt_labels: %d   # pr_labels: %d' %
                               (num_gt_labels,
                                num_pr_labels))
            self._logger.debug('Correspondences: segmentation to gt : '
                               + str([('%d: %d' % (value[0], key)) for (key, value) in
                                      sorted(self._gt2cluster.items(), key=lambda x: x[-1])
                                      if len(value) > 0
                                      ]))
        return

    def levenshtein(self, gt2cluster=None):
        if gt2cluster is None:
            gt2cluster = self._gt2cluster
        levenshteins = []
        max_num_segments = []

        predicted_segments = 0.0
        predicted_segments_non_bg = 0.0

        num_videos = 0

        assert len(self._predicted_labels_per_video) == len(self._gt_labels_per_video)
        background_remapped_labels = set(singleton_lookup(gt2cluster, label)
                                         for label in self._corpus._background_indices
                                         if len(gt2cluster[label]) > 0)
        for ix, (gt_rle, pred_rle) in enumerate(zip(self._gt_rle_per_video, self._predicted_rle_per_video)):
            num_videos += 1
            assert sum(length for _, length in gt_rle) == sum(length for _, length in pred_rle)
            gt_remapped_segments = [singleton_lookup(gt2cluster, label) for (label, length) in gt_rle]
            pred_segments = [label for (label, length) in pred_rle]
            # self._logger.debug("{}: \n\tpred: {}\n\tgold:{}".format(ix, pred_segments, gt_remapped_segments))
            predicted_segments += len(pred_segments)
            predicted_segments_non_bg += len([seg_label for seg_label in pred_segments if seg_label not in background_remapped_labels])
            levenshteins.append(editdistance.eval(gt_remapped_segments, pred_segments))
            max_num_segments.append(max(len(gt_remapped_segments), len(pred_segments)))

        levenshteins = np.array(levenshteins)
        max_num_segments = np.array(max_num_segments)

        assert np.all(max_num_segments > 0)

        results = {
            'mean_levenshtein': np.array([np.mean(levenshteins), 1.0]),
            'mean_max_segments': np.array([np.mean(max_num_segments), 1.0]),
            'total_levenshtein': np.array([np.sum(levenshteins), 1.0]),
            'num_videos': np.array([len(levenshteins), 1.0]),
            'mean_normed_levenshtein': np.array([np.mean(levenshteins / max_num_segments), 1.0]),
            'predicted_segments_per_video': np.array([predicted_segments, num_videos]),
            'predicted_segments_non_bg_per_video': np.array([predicted_segments_non_bg, num_videos]),
        }
        if self._verbose:
            logger.debug("Levenshtein stats")
            for k, v in results.items():
                logger.debug("{}: {}".format(k, v))
        self._return.update(results)

    def single_step_recall(self, gt2cluster=None):
        if gt2cluster is None:
            gt2cluster = self._gt2cluster

        step_match = 0.0
        step_total = 0.0
        non_background_step_match = 0.0
        non_background_step_total = 0.0

        center_step_match = 0.0
        non_background_center_step_match = 0.0

        predicted_label_types = 0.0
        predicted_label_types_non_bg = 0.0
        num_videos = 0.0

        per_action_recalls = {}

        assert len(self._predicted_labels_per_video) == len(self._gt_labels_per_video)
        background_remapped_labels = set(singleton_lookup(gt2cluster, label)
                                         for label in self._corpus._background_indices
                                         if len(gt2cluster[label]) > 0)

        for gt_labels, pred_labels in zip(self._gt_labels_per_video, self._predicted_labels_per_video):
            num_videos += 1
            pred_labels = np.asarray(pred_labels)
            background_timesteps = [lab in self._corpus._background_indices for lab in gt_labels]
            gt_labels_remapped = np.asarray([gt2cluster[gt_label] for gt_label in gt_labels])

            for label in np.unique(pred_labels):
                predicted_label_types += 1
                if label not in background_remapped_labels:
                    predicted_label_types_non_bg += 1

            for label in np.unique(gt_labels_remapped):
                step_total += 1
                if label not in background_remapped_labels:
                    non_background_step_total += 1
                pred_indices = (pred_labels == label).nonzero()[0]
                if label not in per_action_recalls:
                    per_action_recalls[label] = [0,0]
                per_action_recalls[label][1] += 1
                if len(pred_indices) == 0:
                    continue
                pred_index = np.random.choice(pred_indices)
                # center_index = pred_indices[len(pred_indices) // 2]
                center_index = min(pred_indices, key=lambda x:abs(x-(pred_indices[0]+pred_indices[-1])/2))
                if gt_labels_remapped[pred_index] == label:
                    step_match += 1
                    if label not in background_remapped_labels:
                        non_background_step_match += 1
                if gt_labels_remapped[center_index] == label:
                    center_step_match += 1
                    per_action_recalls[label][0] += 1
                    if label not in background_remapped_labels:
                        non_background_center_step_match += 1
        results = ({
            'single_step_recall': np.array([step_match, step_total]),
            'step_recall_non_bg': np.array([non_background_step_match, non_background_step_total]),
            'center_step_recall': np.array([center_step_match, step_total]),
            'center_step_recall_non_bg': np.array([non_background_center_step_match, non_background_step_total]),
            'predicted_label_types_per_video': np.array([predicted_label_types, num_videos]),
            'predicted_label_types_non_bg_per_video': np.array([predicted_label_types_non_bg, num_videos]),
        })
        if self._verbose:
            logger.debug("Single step recall stats")
            for k, v in results.items():
                logger.debug("{}: {}".format(k, v))
            for k, v in per_action_recalls.items():
                log_str = "center_step_recall label {}: {}  ({} / {})".format(k, v[0] / v[1], v[0], v[1])
                if self._corpus is not None:
                    log_str += ' [{}]'.format(self._corpus.index2label[k])
                logger.debug(log_str)
        self._return.update(results)


    def mof(self, optimal_assignment: bool, with_segments=False, optimization='max', possible_gt_labels=None):
        """ Compute mean over frames (MoF) for current labeling.

        Args:
            optimal_assignment: use hungarian to maximize MoF?
            with_segments: if frame sampling was used
            optimization: inside hungarian method
            where: see _create_correspondences method

        Returns:

        """
        self.compute_assignment(optimal_assignment, optimization=optimization, possible_gt_labels=possible_gt_labels)
        if with_segments:
            self._fulfill_segments()
        else:
            self._full_predicted_labels = self.predicted_labels

        background_clusters = [self._gt2cluster[label]  for label in self._corpus._background_indices]

        self._classes_MoF = {}
        self._classes_IoU = {}
        excluded_total = 0
        if self._verbose:
            logger.debug("exclude: {}".format(self.exclude))
        for gt_label in np.unique(self.gt_labels):
            true_defined_frame_n = 0.
            union = 0
            gt_mask = self.gt_labels == gt_label
            # no need the loop since only one label should be here
            # i.e. one-to-one mapping, but i'm lazy
            predicted = 0
            for cluster in self._gt2cluster[gt_label]:
                true_defined_frame_n += np.sum(self._full_predicted_labels[gt_mask] == cluster,
                                               dtype=float)
                pr_mask = self._full_predicted_labels == cluster
                union += np.sum(gt_mask | pr_mask)
                predicted += np.sum(pr_mask)

            self._classes_MoF[gt_label] = [true_defined_frame_n, np.sum(gt_mask)]
            self._classes_IoU[gt_label] = [true_defined_frame_n, union]
            # self._classes_precision[gt_label] = [true_defined_frame_n, predicted]

            if gt_label in self.exclude:
                excluded_total += np.sum(gt_mask)
            else:
                self._frames_true_pr += true_defined_frame_n

        assert len(self.gt_labels_multi) == len(self._full_predicted_labels)

        self._precision = np.zeros(2)
        self._recall = np.zeros(2)

        self._precision_without_bg = np.zeros(2)
        self._recall_without_bg = np.zeros(2)

        self._true_background_frames = np.zeros(2)
        self._pred_background_frames = np.zeros(2)

        self._non_bg_IoU_multi = np.zeros(2)

        self._multiple_labels = np.zeros(2)

        for gt_labels_t, pred_label_t in zip(self.gt_labels_multi, self._full_predicted_labels):
            self._multiple_labels[1] += 1
            if len(gt_labels_t) > 1:
                self._multiple_labels[0] += 1
            gt_clusters_t = [self._gt2cluster[gt_label] for gt_label in gt_labels_t]
            self._recall[1] += len(gt_labels_t)
            self._precision[1] += 1

            true_positive = pred_label_t in gt_clusters_t
            if true_positive:
                self._recall[0] += 1
                self._precision[0] += 1

            self._true_background_frames[1] += 1
            self._pred_background_frames[1] += 1

            pred_background = False
            if pred_label_t in background_clusters:
                self._pred_background_frames[0] += 1
                pred_background = True

            is_background = False
            if any(gt_label in self._corpus._background_indices for gt_label in gt_labels_t):
                is_background = True
                assert all(gt_label in self._corpus._background_indices for gt_label in gt_labels_t)

            if (not is_background) or (not pred_background):
                self._non_bg_IoU_multi[1] += 1
                if true_positive:
                    self._non_bg_IoU_multi[0] += 1

            if is_background:
                self._true_background_frames[0] += 1
            else:
                self._recall_without_bg[1] += len(gt_labels_t)
                self._precision_without_bg[1] += 1
                if pred_label_t in gt_clusters_t:
                    self._recall_without_bg[0] += 1
                    self._precision_without_bg[0] += 1

        self._frames_overall = len(self.gt_labels) - excluded_total
        return self._frames_overall

    def mof_classes(self):
        average_class_mof = 0
        total_true = 0
        total = 0

        average_class_mof_non_bkg = 0
        total_true_non_bkg = 0
        total_non_bkg = 0
        non_bkg_classes = 0
        for key, val in self._classes_MoF.items():
            true_frames, all_frames = val
            # tf_2, pred_frames = self._classes_precision[key]
            # assert tf_2 == true_frames
            if self._verbose:
                log_str = 'mof label %d: %f  %d / %d' % (key, true_frames / all_frames,
                                                        true_frames, all_frames)
                if self._corpus is not None:
                    log_str += '\t[{}]'.format(self._corpus.index2label[key])
                logger.debug(log_str)
                # log_str = 'prec label %d: %f  %d / %d' % (key, tf_2 / pred_frames, tf_2, pred_frames)
                # if self._corpus is not None:
                #     log_str += '\t[{}]'.format(self._corpus.index2label[key])
                # logger.debug(log_str)
            average_class_mof += true_frames / all_frames
            total_true += true_frames
            total += all_frames
            if key not in self._corpus._background_indices:
                non_bkg_classes += 1
                average_class_mof_non_bkg += true_frames / all_frames
                total_true_non_bkg += true_frames
                total_non_bkg += all_frames
        average_class_mof /= len(self._classes_MoF)
        average_class_mof_non_bkg /= non_bkg_classes
        self._return['mof'] = [self._frames_true_pr, self._frames_overall]
        self._return['mof_bg'] = [total_true, total]
        self._return['mof_non_bg'] = [total_true_non_bkg, total_non_bkg]
        self._return['precision'] = self._precision
        self._return['recall'] = self._recall
        if self._precision[1] == 0:
            precision = 0.
        else:
            precision = float(self._precision[0]) / self._precision[1]
        if self._recall[1] == 0:
            recall = 0.
        else:
            recall = float(self._recall[0]) / self._recall[1]
        self._return['f1'] = np.array([(2 * precision * recall) / (precision + recall), 1.0])

        self._return['precision_non_bg'] = self._precision_without_bg
        self._return['recall_non_bg'] = self._recall_without_bg
        if self._precision_without_bg[1] == 0:
            precision_no_bg = 0.
        else:
            precision_no_bg = float(self._precision_without_bg[0]) / self._precision_without_bg[1]
        if self._recall_without_bg[1] == 0:
            recall_no_bg = 0.
        else:
            recall_no_bg = float(self._recall_without_bg[0]) / self._recall_without_bg[1]

        if precision_no_bg == 0 and recall_no_bg == 0:
            f1_no_bg = 0
        else:
            f1_no_bg = (2 * precision_no_bg * recall_no_bg) / (precision_no_bg + recall_no_bg)
        self._return['f1_non_bg'] = np.array([f1_no_bg, 1.0])

        self._return['true_background'] = self._true_background_frames
        self._return['pred_background'] = self._pred_background_frames

        self._return['iou_multi_non_bg'] = self._non_bg_IoU_multi

        self._return['multiple_gt_labels'] = self._multiple_labels

        if self._verbose:
            logger.debug('average class mof: %f' % average_class_mof)
            logger.debug('mof with bg: %f' % (total_true / total))
            logger.debug('average class mof without bg: %f' % average_class_mof_non_bkg)
            logger.debug('mof without bg: %f' % (total_true_non_bkg / total_non_bkg))
            logger.debug('\n')
            logger.debug('f1 with bg: %f' % self._return['f1'][0])
            logger.debug('f1 without bg: %f' % self._return['f1_non_bg'][0])

    def iou_classes(self):
        average_class_iou = 0
        excluded_iou = 0
        non_bg_iou = 0
        for key, val in self._classes_IoU.items():
            true_frames, union = val
            if self._verbose:
                log_str = 'iou label %d: %f  %d / %d' % (key, true_frames / union, true_frames, union)
                if self._corpus is not None:
                    log_str += ' [{}]'.format(self._corpus.index2label[key])
                logger.debug(log_str)
            if key not in self.exclude:
                average_class_iou += true_frames / union
            else:
                excluded_iou += true_frames / union
            if key not in self._corpus._background_indices:
                non_bg_iou += true_frames / union
        average_iou_without_exc = average_class_iou / \
                                  (len(self._classes_IoU) - len(self.exclude))
        average_iou_with_exc = (average_class_iou + excluded_iou) / \
                               len(self._classes_IoU)
        self._return['iou'] = [average_class_iou,
                               len(self._classes_IoU) - len(self.exclude)]
        self._return['iou_bg'] = [average_class_iou + excluded_iou,
                                  len(self._classes_IoU) - len(self.exclude)]
        # TODO: non-bg IOU
        # figure out class averaging
        if self._verbose:
            logger.debug('average IoU: %f' % average_iou_without_exc)
            logger.debug('average IoU with bg: %f' % average_iou_with_exc)


    def mof_val(self):
        if self._verbose:
            self._logger.debug('frames true: %d\tframes overall : %d' %
                               (self._frames_true_pr, self._frames_overall))
        return float(self._frames_true_pr) / self._frames_overall

    def frames(self):
        return self._frames_true_pr

    def stat(self):
        return self._return
