# modified from slim_mallow by Anna Kukleva, https://github.com/Annusha/slim_mallow

import os
import re
from collections import Counter, defaultdict

import numpy as np

from data.features import grouped_pca
from data.corpus import Corpus, GroundTruth, Video, Datasplit
from utils.logger import logger
from utils.utils import all_equal



class BreakfastDatasplit(Datasplit):
    def __init__(self, corpus, remove_background, task_filter=None, splits=None, full=True, subsample=1, feature_downscale=1.0,
                 feature_permutation_seed=None):
        if splits is None:
            splits = list(sorted(BreakfastCorpus.DATASPLITS.keys()))
        self._splits = splits
        self._tasks = BreakfastCorpus.TASKS[:] if task_filter is None else task_filter
        self._p_files = []
        # split
        assert all(split in BreakfastCorpus.DATASPLITS for split in splits)

        for split, p_files in sorted(BreakfastCorpus.DATASPLITS.items()):
            if split in splits:
                assert len(set(p_files) & set(self._p_files)) == 0, "{} : {}".format(set(p_files), set(self._p_files))
                self._p_files.extend(p_files)

        super(BreakfastDatasplit, self).__init__(
            corpus,
            remove_background=remove_background,
            full=full,
            subsample=subsample,
            feature_downscale=feature_downscale,
            feature_permutation_seed=feature_permutation_seed
        )

    def _load_ground_truth_and_videos(self, remove_background):
        self.groundtruth = BreakfastGroundTruth(
            self._corpus,
            task_names=self._tasks,
            p_files=self._p_files,
            remove_background=remove_background,
        )

        k_by_task = {}
        for task, gts in self.groundtruth.gt_by_task.items():
            uniq_labels = set()
            for filename, labels in gts.items():
                uniq_labels = uniq_labels.union(labels_t[0] for labels_t in labels)
            assert -1 not in uniq_labels
            # if -1 in uniq_labels:
            #     k_by_task[task] = len(uniq_labels) - 1
            # else:
            #     k_by_task[task] = len(uniq_labels)
            k_by_task[task] = len(uniq_labels)
        self._K_by_task = k_by_task
        self._init_videos()

    def _init_videos(self):
        # TODO: move to super class?
        gt_stat = Counter()
        video_names = set()
        for root, dirs, files in os.walk(self._corpus._feature_root):
            if files:
                for filename in files:
                    if not filename.endswith(".npy"):
                        continue
                    matching_tasks = [
                        task for task in self._tasks if task in filename
                    ]
                    assert len(matching_tasks) <= 1, "{} matched by {}".format(filename, matching_tasks)
                    if not matching_tasks:
                        continue
                    task = matching_tasks[0]
                    match = re.match(r'(\w*)\.\w*', filename)
                    gt_name = match.group(1)
                    p_name = gt_name.split('_')[0]
                    if p_name not in self._p_files:
                        continue
                    if gt_name not in self.groundtruth.gt_by_task[task]:
                        print("skipping video {} for which no ground truth found!".format(gt_name))
                        continue
                    if not self._full and len(self._videos_by_task[task]) > 10:
                        continue
                    # use extracted features from pretrained on gt embedding
                    # path = os.path.join(root, filename)
                    if self._remove_background:
                        nonbackground_timesteps = self.groundtruth.nonbackground_timesteps_by_task[task][gt_name]
                    else:
                        nonbackground_timesteps = None
                    video = BreakfastVideo(
                        # path,
                        root,
                        remove_background=self._remove_background,
                        nonbackground_timesteps=nonbackground_timesteps,
                        K=self._K_by_task[task],
                        gt=self.groundtruth.gt_by_task[task][gt_name],
                        gt_with_background=self.groundtruth.gt_with_background_by_task[task][gt_name],
                        name=gt_name,
                        cache_features=self._corpus._cache_features,
                        feature_permutation_seed=self._feature_permutation_seed,
                    )
                    # self._features = join_data(self._features, video.features(),
                    #                            np.vstack)

                    # video.reset()  # to not store second time loaded features
                    if task not in self._videos_by_task:
                        self._videos_by_task[task] = {}
                    assert video.name not in self._videos_by_task[task]
                    self._videos_by_task[task][video.name] = video
                    video_names.add(video.name)
                    # accumulate statistic for inverse counts vector for each video
                    gt_stat.update(labels_t[0] for labels_t in self.groundtruth.gt_by_task[task][gt_name])

        logger.debug(
            "{} tasks found with tasks {}, p_files {}".format(len(self._videos_by_task), self._tasks, self._p_files))
        logger.debug("{} videos found with tasks {}, p_files {}".format(len(video_names), self._tasks, self._p_files))

        # # update global range within the current collection for each video
        # for video in self._videos:
        #     video.update_indexes(len(self._features))
        logger.debug('gt statistic: ' + str(gt_stat))
        # FG_MASK
        # self._update_fg_mask()

    # FG_MASK
    # def _update_fg_mask(self):
    #     logger.debug('.')
    #     if self._with_bg:
    #         self._total_fg_mask = np.zeros(len(self._features), dtype=bool)
    #         for video in self._videos:
    #             self._total_fg_mask[np.nonzero(video.global_range)[0][video.fg_mask]] = True
    #     else:
    #         self._total_fg_mask = np.ones(len(self._features), dtype=bool)



class BreakfastCorpus(Corpus):
    BACKGROUND_LABELS = ["SIL"]

    TASKS = [
        'coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake'
    ]

    DATASPLITS = {
        's1': ["P{:02d}".format(d) for d in range(3, 16)],
        's2': ["P{:02d}".format(d) for d in range(16, 29)],
        's3': ["P{:02d}".format(d) for d in range(29, 42)],
        's4': ["P{:02d}".format(d) for d in range(42, 55)],
    }
    assert all_equal(len(v) for v in DATASPLITS.values())

    def __init__(self, mapping_file, feature_root, label_root, task_specific_steps=False):
        self._mapping_file = mapping_file
        self._feature_root = feature_root
        self._label_root = label_root
        self._task_specific_steps = task_specific_steps
        assert not task_specific_steps
        self.annotate_background_with_previous = False

        super(BreakfastCorpus, self).__init__(background_labels=self.BACKGROUND_LABELS)

    def _get_components_for_label(self, label):
        return label.split('_')

    def _load_mapping(self):
        with open(self._mapping_file, 'r') as f:
            for line in f:
                index, label = line.strip().split()
                index = int(index)
                _index = self._index(label)
                if label in self._background_labels:
                    assert index in self._background_indices
                if index in self._background_indices:
                    assert label in self._background_labels
                assert _index == index

    def get_datasplit(self, remove_background, task_filter=None, splits=None, full=True, subsample=1, feature_downscale=1.0,
                      feature_permutation_seed=None):
        return BreakfastDatasplit(self, remove_background, task_filter=task_filter, splits=splits,
                                  full=full, subsample=subsample, feature_downscale=feature_downscale,
                                  feature_permutation_seed=feature_permutation_seed)

def datasets_by_task(mapping_file, feature_root, label_root, remove_background,
                     task_ids=None, splits=BreakfastCorpus.DATASPLITS.keys(), full=True):
    if task_ids is None:
        task_ids = BreakfastCorpus.TASKS
    corpus = BreakfastCorpus(mapping_file, feature_root, label_root)
    return {
        task_id: corpus.get_datasplit(remove_background, [task_id], splits, full)
        for task_id in task_ids
    }

class BreakfastGroundTruth(GroundTruth):

    def __init__(self, corpus, task_names, p_files, remove_background):
        self._p_files = set(p_files)
        super(BreakfastGroundTruth, self).__init__(corpus, task_names, remove_background)

    def _load_gt(self):
        annotation_count = 0
        for root, dirs, files in os.walk(self._corpus._label_root):
            for filename in files:
                if not filename.endswith(".txt"):
                    continue
                p_file = filename.split('_')[0]
                if p_file not in self._p_files:
                    continue
                matching_tasks = [
                    task for task in self._task_names if task in filename
                ]
                assert len(matching_tasks) <= 1, "{} matched by {}".format(filename, matching_tasks)
                if not matching_tasks:
                    continue
                task = matching_tasks[0]

                # ** load labels **
                gt = []
                order = []
                with open(os.path.join(root, filename), 'r') as f:
                    for line in f:
                        match = re.match(r'(\d*)-(\d*)\s*(\w*)', line)
                        start = int(match.group(1))
                        end = int(match.group(2))
                        if end < start:
                            assert match.group(3) == self._corpus.BACKGROUND_LABELS[0]
                            continue
                        assert start > len(gt) - 1
                        label = match.group(3)
                        label_idx = self._corpus._index(label)
                        # gt should be a list of lists, since other corpora can have multiple labels per timestep
                        gt += [[label_idx]] * (end - start + 1)
                        order.append((label_idx, start, end))

                annotation_count += 1

                # ** get vid_name to match feature names **
                up_to_cam, cam_name = os.path.split(root)
                if cam_name == 'stereo':
                    cam_name = 'stereo01'
                _, p_name = os.path.split(up_to_cam)

                match = re.match(r'(\w*)_ch(\d+)\.\w*', filename)
                if match:
                    gt_name = match.group(1)
                    index = int(match.group(2))
                else:
                    match = re.match(r'(\w*)\.\w*', filename)
                    gt_name = match.group(1)
                    index = 0

                # skip videos for which the length of the features and the labels differ by more than 50
                # TODO: get the processed version of the data that fixes this!
                if (gt_name, cam_name) in [
                    ("P51_coffee", "webcam01"),
                    ("P34_coffee", "cam01"),
                    ("P34_juice", "cam01"),
                    ("P52_sandwich", "stereo01"),
                    ("P54_scrambledegg", "webcam01"),
                    ("P34_scrambledegg", "cam01"),
                    ("P34_friedegg", "cam01"),
                    ("P54_pancake", "cam01"),
                    ("P52_pancake", "webcam01"),
                ]:
                    continue

                vid_name = "{}_{}_{}".format(p_name, cam_name, gt_name)

                if task not in self.order_by_task:
                    self.order_by_task[task] = {}
                if task not in self.gt_by_task:
                    self.gt_by_task[task] = {}

                self.gt_by_task[task][vid_name] = gt
                self.order_by_task[task][vid_name] = order
        logger.debug("{} annotation files found".format(annotation_count))

    # def _load_gt(self):
    #     self.gt, self.order = {}, {}
    #     for filename in os.listdir(self.label_root):
    #         if os.path.isdir(os.path.join(self.label_root, filename)):
    #             continue
    #         with open(os.path.join(self.label_root, filename), 'r') as f:
    #             labels = []
    #             local_order = []
    #             curr_lab = -1
    #             start, end = 0, 0
    #             for line in f:
    #                 line = line.split()[0]
    #                 try:
    #                     labels.append(self.label2index[line])
    #                     if curr_lab != labels[-1]:
    #                         if curr_lab != -1:
    #                             local_order.append([curr_lab, start, end])
    #                         curr_lab = labels[-1]
    #                         start = end
    #                     end += 1
    #                 except KeyError:
    #                     break
    #             else:
    #                 # executes every times when "for" wasn't interrupted by break
    #                 self.gt[filename] = np.array(labels)
    #                 # add last labels
    #
    #                 local_order.append([curr_lab, start, end])
    #                 self.order[filename] = local_order


class BreakfastVideo(Video):

    def load_features(self):
        # feats = _features = np.loadtxt(os.path.join(self._feature_root, "{}.txt".format(self.name)))
        feats =  np.load(os.path.join(self._feature_root, "{}.npy".format(self.name)))
        feats = feats[1:, 1:]
        return feats

def extract_feature_groups(corpus):
    group_indices = {
        'reduced_64': (0, 64),
    }
    n_instances = len(corpus)
    grouped = defaultdict(dict)
    for idx in range(n_instances):
        instance = corpus._get_by_index(idx)
        video_name = instance['video_name']
        features = instance['features']
        for group, (start, end) in group_indices.items():
            grouped[group][video_name] = features[:, start:end]
    return grouped

def pca_and_serialize_features(mapping_file, feature_root, label_root, output_feature_root, remove_background,
                               pca_components_per_group=300, by_task=True, task_ids=None):
    all_splits = BreakfastCorpus.DATASPLITS.keys()
    if by_task:
        grouped_datasets = datasets_by_task(mapping_file, feature_root, label_root, remove_background,
                                            task_ids=task_ids, splits=all_splits, full=True)
    else:
        corpus = BreakfastCorpus(mapping_file, feature_root, label_root)
        grouped_datasets = {
            'all': corpus.get_datasplit(remove_background, splits=all_splits)
        }

    os.makedirs(output_feature_root, exist_ok=True)

    for corpora_group, dataset in grouped_datasets.items():
        logger.debug("saving features for task: {}".format(corpora_group))
        grouped_features = extract_feature_groups(dataset)
        transformed, pca_models = grouped_pca(grouped_features, pca_components_per_group, pca_models_by_group=None)
        for feature_group, vid_dict in transformed.items():
            logger.debug("\tsaving features for feature group: {}".format(feature_group))
            feature_group_dir = os.path.join(output_feature_root, feature_group)
            os.makedirs(feature_group_dir, exist_ok=True)
            for vid, features in vid_dict.items():
                fname = os.path.join(feature_group_dir, '{}.npy'.format(vid))
                np.save(fname, features)


if __name__ == "__main__":
    _mapping_file = 'data/breakfast/mapping.txt'
    _feature_root = 'data/breakfast/reduced_fv_64'
    _label_root = 'data/breakfast/BreakfastII_15fps_qvga_sync'

    _components = 64
    for _remove_background in [False, True]:
        for _by_task in [True]:
            _output_feature_root = 'data/breakfast/breakfast_processed/breakfast_pca-{}_{}_{}'.format(
                _components,
                'no-bkg' if _remove_background else 'with-bkg',
                'by-task' if _by_task else 'all-tasks',
            )

            pca_and_serialize_features(_mapping_file, _feature_root, _label_root, _output_feature_root, _remove_background,
                                       pca_components_per_group=_components, by_task=_by_task)
