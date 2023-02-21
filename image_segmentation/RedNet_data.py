import numpy as np
import scipy.io
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
from utils.utils import image_h, image_w

img_dir_train_file = './data/img_dir_train.txt'
depth_dir_train_file = './data/depth_dir_train.txt'
label_dir_train_file = './data/label_train.txt'
img_dir_train_file = './data/img_dir_train.txt'
depth_dir_train_file = './data/depth_dir_train.txt'
label_dir_train_file = './data/label_train.txt'
img_dir_test_file = './data/img_dir_test.txt'
depth_dir_test_file = './data/depth_dir_test.txt'
label_dir_test_file = './data/label_test.txt'
dir_files = "./data/{field}_dir_{split}.txt"


class SUNRGBD(Dataset):
    def __init__(
        self, transform=None, phase="train", data_dir=None,
        train_subset_file=None, val_subset_file=None, test_subset_file=None, debug_mode=False,
    ):
        """
        phase (str): train, val, test
        """

        self.phase = phase
        self.transform = transform
        self.all_dirs = {
            "train": {"img": [], "depth": [], "label": []},
            "val": {"img": [], "depth": [], "label": []},
            "test": {"img": [], "depth": [], "label": []},
        }

        try:
            for dir_split in self.all_dirs:
                for field in self.all_dirs[dir_split]:
                    with open(dir_files.replace("{field}", field).replace("{split}", dir_split), 'r') as f:
                        for line in f:
                            self.all_dirs[dir_split][field].append(line.strip().replace("SUNRGBD_v1//", "SUNRGBD_v1/"))
        except:
            if data_dir is None:
                data_dir = '/path/to/SUNRGB-D'
            SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['trainvalsplit'].train
            split_val = split['trainvalsplit'].val

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            for i, meta in enumerate(SUNRGBDMeta):
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data', data_dir)
                depth_bfx_path = os.path.join(real_dir, 'depth_bfx/' + meta.depthname)
                rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

                label_path = os.path.join(real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel[i][0]]).transpose(1, 0)
                    np.save(label_path, label)

                if meta_dir in split_train:
                    dir_split = "train"
                elif meta_dir in split_val:
                    dir_split = "val"
                else:
                    dir_split = "test"
                self.all_dirs[dir_split]["img"] = np.append(self.all_dirs[dir_split]["img"], rgb_path)
                self.all_dirs[dir_split]["depth"] = np.append(self.all_dirs[dir_split]["depth"], depth_bfx_path)
                self.all_dirs[dir_split]["label"] = np.append(self.all_dirs[dir_split]["label"], label_path)

            local_file_dir = '/'.join(dir_files.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            for dir_split in self.all_dirs:
                for field in self.all_dirs[dir_split]:
                    with open(dir_files.replace("{field}", field).replace("{split}", dir_split), 'w') as f:
                        f.write('\n'.join(self.all_dirs[dir_split][field]))

        self.subsets = {}
        if train_subset_file is not None:
            self.subsets["train"] = set(open(train_subset_file).read().splitlines())
        if val_subset_file is not None:
            self.subsets["val"] = set(open(val_subset_file).read().splitlines())
        if test_subset_file is not None:
            self.subsets["test"] = set(open(test_subset_file).read().splitlines())
        for split in self.subsets:
            # keep items based on whether or not they're in the split...
            subset_img_dir = []
            subset_depth_dir = []
            subset_label_dir = []
            for idx, img_dir in enumerate(self.all_dirs[split]["img"]):
                if img_dir in self.subsets[split]:
                    subset_img_dir.append(img_dir)
                    subset_depth_dir.append(self.all_dirs[split]["depth"][idx])
                    subset_label_dir.append(self.all_dirs[split]["label"][idx])
            self.all_dirs[split]["img"] = subset_img_dir
            self.all_dirs[split]["depth"] = subset_depth_dir
            self.all_dirs[split]["label"] = subset_label_dir

        if debug_mode:
            for split in self.all_dirs:
                for field in self.all_dirs[split]:
                    self.all_dirs[split][field] = self.all_dirs[split][field][:100]

    def __len__(self):
        return len(self.all_dirs[self.phase]["img"])

    def __getitem__(self, idx):
        img_dir = self.all_dirs[self.phase]["img"]
        depth_dir = self.all_dirs[self.phase]["depth"]
        label_dir = self.all_dirs[self.phase]["label"]

        label = np.load(label_dir[idx])
        depth = imageio.imread(depth_dir[idx])
        image = imageio.imread(img_dir[idx])

        sample = {'image': image, 'depth': depth, 'label': label, 'id': img_dir[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'label2': torch.from_numpy(label2).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}
