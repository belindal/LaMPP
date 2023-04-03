# modified from slim_mallow by Anna Kukleva, https://github.com/Annusha/slim_mallow

import os
import copy
import json

import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from evaluation.accuracy import Accuracy
from evaluation.f1 import F1Score
from utils.logger import logger
from utils.utils import nested_dict_map

FEATURE_LABEL_MISMATCH_TOLERANCE = 50

WARN_ON_MISMATCH = False


class Video(object):
    def __init__(self, feature_root, K, remove_background, *, nonbackground_timesteps=None,
                 gt=None, gt_with_background=None, name='', cache_features=False, has_label=True,
                 features_contain_background=True, constraints=None, feature_permutation_seed=None):
        """
        Args:
            feature_root (str): path to video representation
            K (int): number of subactivities in current task
            reset (bool): necessity of holding features in each instance
            gt (arr): ground truth labels
            gt_with_background (arr): ground truth labels with background (0) label
            name (str): short name without any extension
        """
        self.iter = 0
        self._feature_root = feature_root
        self._K = K
        self.name = name
        self._cache_features = cache_features
        self._has_label = has_label
        self._features_contain_background = features_contain_background
        self._constraints = constraints
        self._feature_permutation_seed = feature_permutation_seed

        self._non_background_constraints = None

        assert name

        if remove_background:
            assert has_label
            assert nonbackground_timesteps is not None
            assert len(nonbackground_timesteps) == len(gt)
        self._remove_background = remove_background
        self._nonbackground_timesteps = nonbackground_timesteps

        # self._likelihood_grid = None
        # self._valid_likelihood = None
        # self._theta_0 = 0.1
        # self._subact_i_mask = np.eye(self._K)
        self._features = None
        # self.global_start = start
        # self.global_range = None

        self._n_frames = None

        self._gt = gt if gt is not None else []
        self._gt_unique = np.unique(self._gt)
        self._gt_with_background = gt_with_background

        self._updated_length = False

        # if features is None:
        #     features = self.load_features(self.path)
        # self._set_features(features)

        # self._warned_length = False

        # ordering, init with canonical ordering
        # self._pi = list(range(self._K))
        # self.inv_count_v = np.zeros(self._K - 1)
        # subactivity per frame
        # self._z = []
        # self._z_idx = []
        # self._init_z_framewise()

        # self.fg_mask = np.ones(self.n_frames, dtype=bool)
        # if self._with_bg:
        #     self._init_fg_mask()

        # self._subact_count_update()

        self.segmentation = {'gt': (self._gt, None)}

    def load_features(self):
        raise NotImplementedError("should be implemented by subclasses")

    @property
    def has_label(self):
        return self._has_label

    @property
    def constraints(self):
        if self._remove_background and self._constraints is not None:
            if self._non_background_constraints is None:
                tnb = self._truncated_nonbackground_timesteps()
                constraints = self._constraints[:self.n_frames()]
                constraints = constraints[tnb]
                self._non_background_constraints = constraints
            return self._non_background_constraints
        return self._constraints

    def features(self):
        self._check_truncation()
        if self._cache_features:
            if self._features is None:
                self._features = self._process_features(self.load_features())
            features = self._features
        else:
            features = self._process_features(self.load_features())
        if self._feature_permutation_seed is not None:
            state = np.random.RandomState(self._feature_permutation_seed)
            permutation = np.arange(features.shape[1])
            state.shuffle(permutation)
            features = features[:,permutation]
        return features

    def n_frames(self):
        return self._n_frames

    def _check_truncation(self):
        if not self._has_label:
            return
        n_frames = self.n_frames()
        if n_frames is None:
            # TODO: ugh
            self._process_features(self.load_features())
            n_frames = self.n_frames()
        assert n_frames is not None
        if not self._updated_length and (
                len(self._gt_with_background) != n_frames or not self._features_contain_background):
            self._updated_length = True
            if WARN_ON_MISMATCH:
                print(self.name, '# of gt and # of frames does not match %d / %d' %
                      (len(self._gt_with_background), n_frames))

            assert len(
                self._gt_with_background) - n_frames <= FEATURE_LABEL_MISMATCH_TOLERANCE, "len(self._gt_with_background) = {}, n_frames = {}".format(
                len(self._gt_with_background), n_frames)
            min_n = min(len(self._gt_with_background), n_frames)
            # self._gt = self._gt[:min_n]
            # self._gt_with_background = self._gt_with_background[:min_n]
            self._n_frames = min_n
            # invalidate cache
            self._features = None

    def gt(self):
        self._check_truncation()
        if self._remove_background:
            tnb = self._truncated_nonbackground_timesteps()
            gt = self._gt_with_background[:self.n_frames()]
            new_gt = []
            for ix in tnb:
                new_gt.append(gt[ix])
            gt = new_gt
            assert len(gt) == len(tnb)
        else:
            gt = self._gt[:self.n_frames()]
        return gt

    def gt_with_background(self):
        self._check_truncation()
        return self._gt_with_background[:self.n_frames()]

    def _truncated_nonbackground_timesteps(self):
        return [t for t in self._nonbackground_timesteps if t < self.n_frames()]

    def _process_features(self, features):
        if self._n_frames is None:
            if self._features_contain_background:
                self._n_frames = features.shape[0]
            else:
                self._n_frames = len(self._gt_with_background)
        if not self._features_contain_background:
            return features
        # zeros = 0
        # for i in range(10):
        #     if np.sum(features[:1]) == 0:
        #         features = features[1:]
        #         zeros += 1
        #     else:
        #         break
        # self._gt = self._gt[zeros:]
        # self._gt_with_background = self._gt_with_background[zeros:]
        features = features[:self.n_frames()]
        if self._remove_background:
            features = features[self._truncated_nonbackground_timesteps()]
        return features

    # def _init_z_framewise(self):
    #     """Init subactivities uniformly among video frames"""
    #     # number of frames per activity
    #     step = math.ceil(self.n_frames / self._K)
    #     modulo = self.n_frames % self._K
    #     for action in range(self._K):
    #         # uniformly distribute remainder per actions if n_frames % K != 0
    #         self._z += [action] * (step - 1 * (modulo <= action) * (modulo != 0))
    #     self._z = np.asarray(self._z, dtype=int)
    #     try:
    #         assert len(self._z) == self.n_frames
    #     except AssertionError:
    #         logger.error('Wrong initialization for video %s', self.path)

    # FG_MASK
    # def _init_fg_mask(self):
    #     indexes = [i for i in range(self.n_frames) if i % 2]
    #     self.fg_mask[indexes] = False
    #     # todo: have to check if it works correctly
    #     # since it just after initialization
    #     self._z[self.fg_mask == False] = -1

    # def update_indexes(self, total):
    #     self.global_range = np.zeros(total, dtype=bool)
    #     self.global_range[self.global_start: self.global_start + self.n_frames] = True

    # def reset(self):
    #     """If features from here won't be in use anymore"""
    #     self._features = None

    # def z(self, pi=None):
    #     """Construct z (framewise label assignments) from ordering and counting.
    #     Args:
    #         pi: order, if not given the current one is used
    #     Returns:
    #         constructed z out of indexes instead of actual subactivity labels
    #     """
    #     self._z = []
    #     self._z_idx = []
    #     if pi is None:
    #         pi = self._pi
    #     for idx, activity in enumerate(pi):
    #         self._z += [int(activity)] * self.a[int(activity)]
    #         self._z_idx += [idx] * self.a[int(activity)]
    #     if opt.bg:
    #         z = np.ones(self.n_frames, dtype=int) * -1
    #         z[self.fg_mask] = self._z
    #         self._z = z[:]
    #         z[self.fg_mask] = self._z_idx
    #         self._z_idx = z[:]
    #     assert len(self._z) == self.n_frames
    #     return np.asarray(self._z_idx)


class Datasplit(Dataset):
    def __init__(self, corpus, remove_background, full=True, subsample=1, feature_downscale=1.0,
                 feature_permutation_seed=None):
        self._corpus = corpus
        self._remove_background = remove_background
        self._full = full

        self._feature_permutation_seed = feature_permutation_seed

        # logger.debug('%s  subactions: %d' % (subaction, self._K))
        self.return_stat = {}

        self._videos_by_task = {}
        # init with ones for consistency with first measurement of MoF
        # self._subact_counter = np.ones(self._K)
        # number of gaussian in each mixture
        self._gt2label = None
        self._label2gt = {}

        # FG_MASK
        # self._total_fg_mask = None

        # multiprocessing for sampling activities for each video
        self._features_by_task = None
        self._embedded_feat = None

        self.groundtruth = None
        self._K_by_task = None
        self._load_ground_truth_and_videos(remove_background)
        assert self.groundtruth is not None, "_load_ground_truth_and_videos didn't set groundtruth"
        assert len(self._videos_by_task) != 0, "_load_ground_truth_and_videos didn't load any task's videos"
        assert self._K_by_task is not None, "_load_ground_truth_and_videos didn't set _K_by_task"

        self._tasks_and_video_names = list(sorted([
            (task_name, video_name)
            for task_name, vid_dict in self._videos_by_task.items()
            for video_name in vid_dict
        ]))

        self.subsample = subsample

        self.feature_downscale = feature_downscale

        # logger.debug('min: %f  max: %f  avg: %f' %
        #              (np.min(self._features),
        #               np.max(self._features),
        #               np.mean(self._features)))

    def batch_sampler(self, batch_size=1, batch_by_task=True, shuffle=False):
        return BatchSampler(self, batch_size=batch_size, batch_by_task=batch_by_task, shuffle=shuffle)

    @property
    def corpus(self):
        return self._corpus

    @property
    def remove_background(self):
        return self._remove_background

    def __len__(self):
        return len(self._tasks_and_video_names)

    def __getitem__(self, task_and_video_name, wrap_torch=True):
        task_name, video_name = task_and_video_name
        video_obj: Video = self._videos_by_task[task_name][video_name]

        # num_timesteps = torch_features.size(0)
        try:
            features = video_obj.features()
        except Exception as e:
            print("exception with task and video {}".format(task_and_video_name))
            print(e)
            return None
        task_indices = self.corpus.indices_by_task(task_name)
        if self.remove_background:
            task_indices = set(task_indices) - set(self.corpus._background_indices)
        task_indices = sorted(task_indices)
        if video_obj.has_label:
            gt_single = [gt_t[0] for gt_t in video_obj.gt()]

        constraints = video_obj.constraints

        if self.subsample != 1:
            subsample_indices = np.arange(features.shape[0] // self.subsample) * self.subsample
            subsample_boundaries = list(
                zip(list(subsample_indices), list(subsample_indices - 1)[1:] + [features.shape[0] - 1]))
            if video_obj.has_label:
                gt_single_sampled = list(np.array(gt_single)[subsample_indices])
            features = features[subsample_indices]
        else:
            subsample_indices = np.arange(features.shape[0])
            subsample_boundaries = list(zip(subsample_indices, subsample_indices))
            if video_obj.has_label:
                gt_single_sampled = gt_single

        if wrap_torch:
            features = torch.from_numpy(features).float()
            task_indices = torch.LongTensor(task_indices)
            if video_obj._has_label:
                gt_single = torch.LongTensor(gt_single)
                gt_single_sampled = torch.LongTensor(gt_single_sampled)
            if constraints is not None:
                constraints = torch.from_numpy(constraints).float()
        else:
            task_indices = list(task_indices)

        if self.feature_downscale != 1.0:
            features = features / self.feature_downscale
        data = {
            'task_name': task_name,
            'video_name': video_name,
            'features': features,
            'task_indices': task_indices,
            'subsample_indices': subsample_indices,
            'subsample_boundaries': subsample_boundaries,
        }

        if constraints is not None:
            data['constraints'] = constraints

        if video_obj._has_label:
            data.update({
                'gt': video_obj.gt(),
                'gt_single_unsampled': gt_single,
                'gt_single': gt_single_sampled,
                'gt_with_background': video_obj.gt_with_background(),
            })
        return data

    def _get_by_index(self, index, wrap_torch=True):
        task_name, video_name = self._tasks_and_video_names[index]
        return self.__getitem__((task_name, video_name), wrap_torch)

    @property
    def feature_dim(self):
        return self._get_by_index(0)['features'].size(1)

    def _load_ground_truth_and_videos(self, remove_background):
        raise NotImplementedError("subclasses should implement _load_ground_truth")

    def get_allowed_starts_and_transitions(self):
        raise NotImplementedError("subclasses should implement get_allowed_starts_and_transitions")

    def get_ordered_indices_no_background(self):
        raise NotImplementedError("subclasses should implement get_allowed_starts_and_transitions")

    def canonicalize_background(self, index):
        if index in self._corpus._background_indices:
            return self._corpus._background_indices[0]
        else:
            return index

    def accuracy_corpus(self, optimal_assignment: bool, prediction_function, prefix='', verbose=True,
                        compare_to_folder=None):
        """Calculate metrics as well with previous correspondences between
        gt labels and output labels"""
        stats_by_task = {}

        if compare_to_folder is not None:
            task_mapping = {}
            if os.path.exists(os.path.join(compare_to_folder, "y_true.json")):
                with open(os.path.join(compare_to_folder, "y_true.json")) as f:
                    y_true_all = json.load(f)
                with open(os.path.join(compare_to_folder, "y_pred.json")) as f:
                    y_pred_all = json.load(f)
            else:
                y_true_all = None
                y_pred_all = None

        for task in self._videos_by_task:
            if verbose:
                logger.debug("computing accuracy for task {}".format(task))
            accuracy = Accuracy(verbose=verbose, corpus=self._corpus)

            f1_score = F1Score(K=self._K_by_task[task], n_videos=len(self._videos_by_task[task]), verbose=verbose)
            long_gt = []
            long_pr = []

            if compare_to_folder is not None:
                compare_accuracy = Accuracy(verbose=verbose, corpus=self._corpus)
                compare_long_gt = []
                compare_long_pr = []

            # long_gt_onhe0 = []
            self.return_stat = {}

            def load_predictions(video_name):
                if y_true_all is not None:
                    return {
                        'y_true': np.array(y_true_all[str(task)][video_name]),
                        'y_pred': np.array(y_pred_all[str(task)][video_name]),
                    }
                elif os.path.exists(os.path.join(compare_to_folder, "{}_y_true.npy".format(video_name))):
                    y_true = np.load(os.path.join(compare_to_folder, "{}_y_true.npy".format(video_name)))
                    y_pred = np.load(os.path.join(compare_to_folder, "{}_y_pred.npy".format(video_name)))
                    return {
                        'y_true': y_true,
                        'y_pred': y_pred,
                    }
                else:
                    with open(os.path.join(compare_to_folder, "{}.json".format(video_name))) as f:
                        pred_data = json.load(f)
                        return {
                            key: np.array(val)
                            for key, val in pred_data.items()
                        }

            for video_name, video in self._videos_by_task[task].items():
                # long_gt += list(video._gt_with_0)
                # long_gt_onhe0 += list(video._gt)
                gt = list(video.gt())
                if prediction_function is not None:
                    pred = list(prediction_function(video))
                    if self.subsample != 1:
                        # _data = self[(task, video_name)]
                        # boundaries = _data['subsample_boundaries']
                        # pred = list(accuracy._fulfill_segments_nondes(boundaries, pred, len(gt)))
                        pred = list(np.array(pred + [pred[-1]]).repeat(self.subsample)[:len(gt)])

                        assert len(gt) == len(pred), "{} != {}".format(len(gt), len(pred))

                    if self.corpus.annotate_background_with_previous:
                        gt = [
                            [self.canonicalize_background(ix) for ix in gt_t]
                            for gt_t in gt
                        ]
                        pred = [self.canonicalize_background(ix) for ix in pred]
                        # print("video: {}".format(video_name))
                        # print("gt: {}".format([gt_t[0] for gt_t in gt]))
                        # print("pred: {}".format(pred))
                        # print("gt enum: {}".format(list(enumerate([gt_t[0] for gt_t in gt]))))
                        # print("pred enum: {}".format(list(enumerate(pred))))
                        # print()

                    accuracy.add_gt_labels(gt)
                    accuracy.add_predicted_labels(pred)
                    long_gt += gt
                    long_pr += pred

                if compare_to_folder is not None:
                    # break ties with background, or earlier steps
                    pred_data = load_predictions(video_name)
                    y_true = pred_data['y_true']
                    y_pred = pred_data['y_pred']
                    trues = y_true.argmax(axis=1)
                    preds = y_pred.argmax(axis=1)

                    assert len(trues) == len(video.gt())
                    for t, g in zip(trues, video.gt()):
                        g = g[0]
                        if t in task_mapping:
                            assert task_mapping[t] == g
                        else:
                            task_mapping[t] = g

            if compare_to_folder is not None:
                for video_name, video in self._videos_by_task[task].items():
                    pred_data = load_predictions(video_name)
                    y_true = pred_data['y_true']
                    y_pred = pred_data['y_pred']
                    # break ties with background, or earlier steps
                    trues = y_true.argmax(axis=1)
                    preds = y_pred.argmax(axis=1)
                    # y_true_rc = pred_data['y_true_rc']
                    # y_pred_rc = pred_data['y_pred_rc']
                    # trues = y_true_rc.argmax(axis=1)
                    # preds = y_pred_rc.argmax(axis=1)

                    gt = [[task_mapping[t]] for t in trues]
                    pred = [task_mapping[p] for p in preds]

                    compare_accuracy.add_gt_labels(gt)
                    compare_accuracy.add_predicted_labels(pred)

            named_accuracies = []
            if prediction_function is not None:
                named_accuracies.append(('model', accuracy))
                accuracy_to_return = accuracy
            else:
                accuracy_to_return = compare_accuracy
            if compare_to_folder is not None:
                named_accuracies.append(('comparison: {}'.format(compare_to_folder), compare_accuracy))

            for acc_name, acc in named_accuracies:
                # if opt.bg:
                #     # enforce bg class to be bg class
                #     acc.exclude[-1] = [-1]
                # if not opt.zeros and opt.dataset == 'bf': #''Breakfast' in opt.dataset_root:
                #     # enforce to SIL class assign nothing
                #     acc.exclude[0] = [-1]

                if verbose:
                    logger.debug('Stats for {}'.format(acc_name))

                total_fr = acc.mof(optimal_assignment, possible_gt_labels=self.corpus.indices_by_task(task))
                if acc_name == 'model':
                    self._gt2label = acc._gt2cluster
                    self._label2gt = {}
                    for key, val in self._gt2label.items():
                        try:
                            self._label2gt[val[0]] = key
                        except IndexError:
                            pass
                acc_cur = acc.mof_val()
                if verbose:
                    logger.debug('%s Task: %s' % (prefix, task))
                    logger.debug('%s MoF val: ' % prefix + str(acc_cur))
                    # logger.debug('%s previous dic -> MoF val: ' % prefix + str(float(old_mof) / total_fr))

                acc.mof_classes()
                acc.iou_classes()
                acc.levenshtein()
                acc.single_step_recall()
                if verbose:
                    logger.debug('\n')

            self.return_stat = accuracy_to_return.stat()

            if prediction_function is not None:
                f1_score.set_gt(long_gt)
                f1_score.set_pr(long_pr)
                f1_score.set_gt2pr(self._gt2label)
                # if opt.bg:
                #     f1_score.set_exclude(-1)
                f1_score.f1()

                for key, val in f1_score.stat().items():
                    self.return_stat[key] = val

                for video_name, video in self._videos_by_task[task].items():
                    video.segmentation[video.iter] = (prediction_function(video), self._label2gt)

            stats = accuracy_to_return.stat()
            stats['num_videos'] = np.array([len(self._videos_by_task[task]), 1])
            if compare_to_folder is not None:
                comparison_stats = compare_accuracy.stat()
                if verbose:
                    logger.debug("\n")
                stats['comparison_mof'] = comparison_stats['mof']
                stats['comparison_mof_bg'] = comparison_stats['mof_bg']
                stats['comparison_mof_non_bg'] = comparison_stats['mof_non_bg']
                stats['comparison_step_recall_non_bg'] = comparison_stats['step_recall_non_bg']
                stats['comparison_mean_normed_levenshtein'] = comparison_stats['mean_normed_levenshtein']

                stats['comparison_f1'] = comparison_stats['f1']
                stats['comparison_f1_non_bg'] = comparison_stats['f1_non_bg']
                stats['comparison_center_step_recall_non_bg'] = comparison_stats['step_recall_non_bg']

                stats['comparison_pred_background'] = comparison_stats['pred_background']

            stats_by_task[task] = accuracy_to_return.stat()
        return stats_by_task

    # def resume_segmentation(self):
    #     for video in self._videos:
    #         video.iter = self.iter
    #         video.resume()
    #     self._count_subact()


class BatchSampler(Sampler):
    def __init__(self, datasplit: Datasplit, batch_size: int, batch_by_task: bool, shuffle: bool, seed=1):
        self.datasplit = datasplit
        self.batch_size = batch_size
        self.batch_by_task = batch_by_task
        self.shuffle = shuffle
        self.seed = seed

        self.batches = []
        if shuffle:
            self.random_state = random.Random(self.seed)
        else:
            self.random_state = None
        task_names = list(sorted(self.datasplit._videos_by_task.keys()))

        videos_by_task = {
            task: list(sorted(videos))
            for task, videos in self.datasplit._videos_by_task.items()
        }

        for task in task_names:
            videos = videos_by_task[task]
            for i in range(0, len(videos), self.batch_size):
                self.batches.append([(task, video) for video in videos[i:i + self.batch_size]])

    def __iter__(self):
        if self.random_state is not None:
            self.random_state.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class Corpus(object):

    def __init__(self, background_labels, cache_features=False):
        """
        Args:
            K: number of possible subactions in current dataset (TODO: this disappeared)
            subaction: current name of complex activity
        """
        self.label2index = {}
        self.index2label = {}
        self.component2index = {}
        self.index2component = {}

        self.label_indices2component_indices = {}

        self._cache_features = cache_features

        self._labels_frozen = False
        self._background_labels = background_labels
        self._background_indices = []
        for label in background_labels:
            self._background_indices.append(self._index(label))
        self._indices_by_task = {}
        self._load_mapping()
        self._labels_frozen = True

    @property
    def n_classes(self):
        return len(self.label2index)

    @property
    def n_components(self):
        return len(self.component2index)

    def _index(self, label):
        if label not in self.label2index:
            assert not self._labels_frozen, "trying to index {} after index has been frozen".format(label)
            label_idx = len(self.label2index)
            self.label2index[label] = label_idx
            self.index2label[label_idx] = label
            assert label not in self.label_indices2component_indices
            component_indices = []
            for component_label in self._get_components_for_label(label):
                component_idx = self._index_component(component_label)
                component_indices.append(component_idx)
            self.label_indices2component_indices[label_idx] = list(sorted(component_indices))
        else:
            label_idx = self.label2index[label]
        return label_idx

    def _index_component(self, component_label):
        if component_label not in self.component2index:
            assert not self._labels_frozen, "trying to index COMPONENT {} after index has been frozen".format(
                component_label)
            component_idx = len(self.component2index)
            self.component2index[component_label] = component_idx
            self.index2component[component_idx] = component_label
        else:
            component_idx = self.component2index[component_label]
        return component_idx

    def _get_components_for_label(self, label):
        raise NotImplementedError()

    def indices_by_task(self, task):
        return list(sorted(self._indices_by_task[task]))

    def update_indices_by_task(self, task, indices):
        if task not in self._indices_by_task:
            self._indices_by_task[task] = set()
        self._indices_by_task[task].update(indices)

    def _load_mapping(self):
        raise NotImplementedError("subclasses should implement _load_mapping")

    def get_datasplit(self, remove_background, full=True) -> Datasplit:
        raise NotImplementedError("subclasses should implement get_datasplit")

    # def _count_subact(self):
    #     self._subact_counter = np.zeros(self._K)
    #     for video in self._videos:
    #         self._subact_counter += video.a


class GroundTruth(object):
    def __init__(self, corpus: Corpus, task_names, remove_background):
        self._corpus = corpus
        self._task_names = task_names
        self._remove_background = remove_background

        self.gt_by_task = {}
        self.gt_with_background_by_task = {}
        self.order_by_task = {}
        self.order_with_background_by_task = {}

        self.nonbackground_timesteps_by_task = {}
        self.load_gt_and_remove_background()

    def _load_gt(self):
        raise NotImplementedError("_load_gt")

    def load_gt_and_remove_background(self):
        self._load_gt()
        self.gt_with_background_by_task = self.gt_by_task
        # print(list(gt_with_0.keys()))
        self.order_with_background_by_task = self.order_by_task

        if self._remove_background:
            self.remove_background()

        for task, gt_dict in self.gt_by_task.items():
            label_set = set()
            for vid, gt in gt_dict.items():
                for gt_t in gt:
                    label_set.update(gt_t)
            self._corpus.update_indices_by_task(task, label_set)

    def remove_background(self):
        self.gt_with_background_by_task = copy.deepcopy(self.gt_by_task)
        self.order_with_background_by_task = copy.deepcopy(self.order_by_task)

        def nonbkg_indices(task, video, gt):
            return [t for t, gt_t in enumerate(gt) if gt_t[0] not in self._corpus._background_indices]

        self.nonbackground_timesteps_by_task = nested_dict_map(self.gt_by_task, nonbkg_indices)

        def rm_bkg_from_indices(task, video, gt):
            nonbackground_indices = self.nonbackground_timesteps_by_task[task][video]
            nbi_set = set(nonbackground_indices)
            new_gt = []
            for ix, val in enumerate(gt):
                if ix in nbi_set:
                    new_gt.append(val)
            gt = new_gt
            # if value[0] == 0:
            #     for idx, val in enumerate(value):
            #         if val:
            #             value[:idx] = val
            #             break
            # if value[-1] == 0:
            #     for idx, val in enumerate(np.flip(value, 0)):
            #         if val:
            #             value[-idx:] = val
            #             break
            assert not any(ix in gt for ix in self._corpus._background_indices)
            return gt

        def rm_bkg_from_order(task, video, order):
            return [t for t in order if t[0] not in self._corpus._background_indices]

        self.gt_by_task = nested_dict_map(self.gt_by_task, rm_bkg_from_indices)
        self.order_by_task = nested_dict_map(self.order_by_task, rm_bkg_from_order)

    # def sparse_gt(self):
    #     for key, val in self.gt.items():
    #         sparse_segm = [i for i in val[::10]]
    #         self.gt[key] = sparse_segm
    #     self.gt_with_background = copy.deepcopy(self.gt)
