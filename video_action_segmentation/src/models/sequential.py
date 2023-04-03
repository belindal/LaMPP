import tqdm
import numpy as np
import torch
import torch.nn as nn
from models.model import Model, make_optimizer, make_data_loader
from utils.utils import all_equal

from data.corpus import Datasplit


class Encoder(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--seq_num_layers', type=int, default=2)

    def __init__(self, args, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.args = args
        assert output_dim % 2 == 0
        # TODO: dropout?
        self.encoder = nn.LSTM(input_dim, output_dim // 2, bidirectional=True, num_layers=args.seq_num_layers, batch_first=True)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()

    def forward(self, features, lengths, output_padding_value=0):
        packed = nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        encoded_packed, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded_packed, batch_first=True, padding_value=output_padding_value)
        return encoded

class SequentialPredictConstraints(Model):
    @classmethod
    def add_args(cls, parser):
        pass

    @classmethod
    def from_args(cls, args, train_data: Datasplit):
        return cls(args, train_data)


    def __init__(self, args, train_data: Datasplit):
        from data.crosstask import CrosstaskDatasplit
        assert isinstance(train_data, CrosstaskDatasplit)

        self.args = args
        self.n_classes = train_data._corpus.n_classes
        self.remove_background = train_data.remove_background

        self.ordered_nonbackground_indices_by_task = {
            task_id: [train_data.corpus._index(step) for step in task.steps]
            for task_id, task in train_data._tasks_by_id.items()
        }

        self.background_indices_by_task = {
            task_id: list(sorted(ix for ix in train_data.corpus.indices_by_task(task_id)
                                 if ix in set(train_data.corpus._background_indices)))
            for task_id in train_data._tasks_by_id.keys()
        }
        assert all(len(v) == 1 for v in self.background_indices_by_task.values()), self.background_indices_by_task

        if train_data.remove_background:
            self.canonical = SequentialCanonicalBaseline(args, train_data)
        else:
            self.canonical = None

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        pass

    def predict(self, test_data: Datasplit):
        predictions = {}
        loader = make_data_loader(self.args, test_data, batch_by_task=False, shuffle=False, batch_size=1)


        for batch in loader:
            features = batch['features'].squeeze(0)
            num_timesteps = features.size(0)

            tasks = batch['task_name']
            assert len(tasks) == 1
            task = next(iter(tasks))
            videos = batch['video_name']
            assert len(videos) == 1
            video = next(iter(videos))

            # constraints: T x K
            constraints = batch['constraints'].squeeze(0)
            assert constraints.size(0) == num_timesteps

            step_indices = self.ordered_nonbackground_indices_by_task[task]
            background_indices = self.background_indices_by_task[task]

            active_step = constraints.argmax(dim=1)
            active_step.apply_(lambda ix: step_indices[ix])
            if not test_data.remove_background:
                active_step[constraints.sum(dim=1) == 0] = background_indices[0]
                predictions[video] = active_step.cpu().numpy()
            else:
                preds = active_step.cpu().numpy()
                zero_indices = (constraints.sum(dim=1) == 0).nonzero().flatten()
                baseline_preds = self.canonical.predict_single(task, num_timesteps)
                for ix in zero_indices:
                    preds[ix] = baseline_preds[ix]
                predictions[video] = preds
                # just arbitrarily choose a background index, they will get canonicalized anyway
        return predictions

class SequentialGroundTruth(Model):
    @classmethod
    def add_args(cls, parser):
        pass

    @classmethod
    def from_args(cls, args, train_data: Datasplit):
        return cls(args, train_data)

    def __init__(self, args, train_data: Datasplit):
        from data.crosstask import CrosstaskDatasplit
        assert isinstance(train_data, CrosstaskDatasplit)
        self.args = args
        self.n_classes = train_data._corpus.n_classes
        self.remove_background = train_data.remove_background

        pass

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        pass

    def predict(self, test_data: Datasplit):
        predictions = {}
        loader = make_data_loader(self.args, test_data, batch_by_task=False, shuffle=False, batch_size=1)

        for batch in loader:
            features = batch['features'].squeeze(0)
            # num_timesteps = features.size(0)

            tasks = batch['task_name']
            assert len(tasks) == 1
            # task = next(iter(tasks))
            videos = batch['video_name']
            assert len(videos) == 1
            video = next(iter(videos))

            predictions[video] = batch['gt_single'].squeeze(0).numpy().tolist()
        return predictions

class SequentialCanonicalBaseline(Model):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--canonical_baseline_background_fraction', type=float, default=0.0)

    @classmethod
    def from_args(cls, args, train_data: Datasplit):
        return cls(args, train_data)

    def __init__(self, args, train_data: Datasplit):
        from data.crosstask import CrosstaskDatasplit
        assert isinstance(train_data, CrosstaskDatasplit)
        self.args = args
        self.n_classes = train_data._corpus.n_classes
        self.remove_background = train_data.remove_background

        self.ordered_nonbackground_indices_by_task = {
            task_id: [train_data.corpus._index(step) for step in task.steps]
            for task_id, task in train_data._tasks_by_id.items()
        }

        self.background_indices_by_task = {
            task_id: list(sorted(ix for ix in train_data.corpus.indices_by_task(task_id)
                      if ix in set(train_data.corpus._background_indices)))
            for task_id in train_data._tasks_by_id.keys()
        }
        assert all(len(v) == 1 for v in self.background_indices_by_task.values()), self.background_indices_by_task

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        pass

    def predict_single(self, task_id, num_timesteps):
        if self.remove_background:
            num_background_frames = 0
        else:
            num_background_frames = int(num_timesteps * self.args.canonical_baseline_background_fraction)
            background_index = next(iter(self.background_indices_by_task[task_id]))

        nonbackground_indices = self.ordered_nonbackground_indices_by_task[task_id]

        # this fails if we've removed background b/c some videos are too short
        if not self.remove_background:
            assert num_timesteps >= len(nonbackground_indices)

        # expand the total nonbackground duration to fit all background frames
        num_nonbackground_frames = max(num_timesteps - num_background_frames, len(nonbackground_indices))

        step_duration = num_nonbackground_frames // len(nonbackground_indices)
        assert step_duration >= 1

        if self.remove_background or num_background_frames == 0:
            background_duration = 0
            pad = nonbackground_indices[-1]
        else:
            background_duration = (num_timesteps - step_duration * len(nonbackground_indices)) // (len(nonbackground_indices) + 1)
            assert background_duration >= 0
            pad = background_index

        indices = []
        for step_ix in nonbackground_indices:
            if not self.remove_background:
                indices.extend([background_index] * background_duration)
            indices.extend([step_ix] * step_duration)

        if not self.remove_background:
            assert len(indices) <= num_timesteps
        assert num_timesteps - len(indices) - background_duration <= len(nonbackground_indices) + 1
        indices.extend([pad] * (num_timesteps - len(indices)))
        # hack for remove_background case: some videos have e.g. only 6 frames for 8 steps
        indices = indices[:num_timesteps]
        return indices

    def predict(self, test_data: Datasplit):
        predictions = {}
        loader = make_data_loader(self.args, test_data, batch_by_task=False, shuffle=False, batch_size=1)

        for batch in loader:
            features = batch['features'].squeeze(0)
            num_timesteps = features.size(0)

            tasks = batch['task_name']
            assert len(tasks) == 1
            task = next(iter(tasks))
            videos = batch['video_name']
            assert len(videos) == 1
            video = next(iter(videos))

            predictions[video] = self.predict_single(task, num_timesteps)
        return predictions

class SequentialPredictFrames(nn.Module):
    @classmethod
    def add_args(cls, parser):
        Encoder.add_args(parser)
        parser.add_argument('--seq_hidden_size', type=int, default=200)

    def __init__(self, args, input_dim, num_classes):
        super(SequentialPredictFrames, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.encoder = Encoder(self.args, input_dim, args.seq_hidden_size)
        self.proj = nn.Linear(args.seq_hidden_size, num_classes)

    def forward(self, features, lengths, valid_classes_per_instance=None):
        # batch_size x max_len x seq_hidden_size
        encoded = self.encoder(features, lengths, output_padding_value=0)
        # batch_size x max_len x num_classes
        logits = self.proj(encoded)
        if valid_classes_per_instance is not None:
            assert all_equal(set(vc.detach().cpu().numpy()) for vc in
                             valid_classes_per_instance), "must have same valid_classes for all instances in the batch"
            valid_classes = valid_classes_per_instance[0]
            mask = torch.full_like(logits, -float("inf"))
            mask[:,:,valid_classes] = 0
            logits = logits + mask
        return logits

class SequentialDiscriminative(Model):
    @classmethod
    def add_args(cls, parser):
        SequentialPredictFrames.add_args(parser)

    @classmethod
    def from_args(cls, args, train_data: Datasplit):
        return cls(args, train_data)

    def __init__(self, args, train_data: Datasplit):
        self.args = args
        #self.n_classes = sum(len(indices) for indices in train_data.groundtruth.indices_by_task.values())
        self.n_classes = train_data._corpus.n_classes
        self.model = SequentialPredictFrames(args, input_dim=train_data.feature_dim, num_classes=self.n_classes)
        if args.cuda:
            self.model.cuda()

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        assert use_labels
        IGNORE = -100
        loss = nn.CrossEntropyLoss(ignore_index=IGNORE)
        optimizer, scheduler = make_optimizer(self.args, self.model.parameters())
        loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=True, batch_size=self.args.batch_size)

        for epoch in range(self.args.epochs):
            # call here since we may set eval in callback_fn
            self.model.train()
            losses = []
            assert self.args.batch_accumulation <= 1
            for batch in tqdm.tqdm(loader, ncols=80):
            # for batch in loader:
                tasks = batch['task_name']
                videos = batch['video_name']
                features = batch['features']
                gt_single = batch['gt_single']
                task_indices = batch['task_indices']
                max_len = features.size(1)
                lengths = batch['lengths']
                invalid_mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
                if self.args.cuda:
                    features = features.cuda()
                    lengths = lengths.cuda()
                    task_indices = [indx.cuda() for indx in task_indices]
                    gt_single = gt_single.cuda()
                    invalid_mask = invalid_mask.cuda()
                gt_single.masked_fill_(invalid_mask, IGNORE)
                # batch_size x max_len x num_classes
                logits = self.model(features, lengths, valid_classes_per_instance=task_indices)

                this_loss = loss(logits.view(-1, logits.size(-1)), gt_single.flatten())
                losses.append(this_loss.item())
                this_loss.backward()

                optimizer.step()
                self.model.zero_grad()
            train_loss = np.mean(losses)
            if scheduler is not None:
                scheduler.step(train_loss)
            callback_fn(epoch, {'train_loss': train_loss})

            # if evaluate_on_data_fn is not None:
            #     train_mof = evaluate_on_data_fn(self, train_data, 'train')
            #     dev_mof = evaluate_on_data_fn(self, dev_data, 'dev')
            #     dev_mof_by_epoch[epoch] = dev_mof
            #     log_str += ("\ttrain mof: {:.4f}".format(train_mof))
            #     log_str += ("\tdev mof: {:.4f}".format(dev_mof))

    def predict(self, test_data: Datasplit):
        self.model.eval()
        predictions = {}
        loader = make_data_loader(self.args, test_data, batch_by_task=False, shuffle=False, batch_size=1)
        for batch in loader:
            features = batch['features']
            lengths = batch['lengths']
            task_indices = batch['task_indices']
            if self.args.cuda:
                features = features.cuda()
                lengths = lengths.cuda()
                task_indices = [indx.cuda() for indx in task_indices]
            videos = batch['video_name']
            assert all_equal(videos)
            video = next(iter(videos))
            # batch_size x length x num_classes
            with torch.no_grad():
                logits = self.model(features, lengths, valid_classes_per_instance=task_indices)
                preds = logits.max(dim=-1)[1]
                preds = preds.squeeze(0)
                assert preds.ndim == 1
                predictions[video] = preds.detach().cpu().numpy()
        return predictions

