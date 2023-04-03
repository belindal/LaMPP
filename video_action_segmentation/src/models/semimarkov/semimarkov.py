import copy
import numpy as np
import tqdm
import time

import torch
from data.corpus import Datasplit
from models.model import Model, make_optimizer, make_data_loader
from models.semimarkov.semimarkov_modules import SemiMarkovModule, ComponentSemiMarkovModule
from models.semimarkov import semimarkov_utils

from utils.utils import all_equal


class SemiMarkovModel(Model):
    @classmethod
    def add_args(cls, parser):
        SemiMarkovModule.add_args(parser)
        ComponentSemiMarkovModule.add_args(parser)
        parser.add_argument('--sm_component_model', action='store_true')

        parser.add_argument('--sm_constrain_transitions', action='store_true')

        parser.add_argument('--sm_constrain_with_narration', choices=['train', 'test'], nargs='*', default=[])
        parser.add_argument('--sm_constrain_narration_weight', type=float, default=-1e4)

        parser.add_argument('--sm_train_discriminatively', action='store_true')

        parser.add_argument('--sm_hidden_markov', action='store_true', help='train as hidden markov model (fix K=1) and length distribution')

        parser.add_argument('--sm_predict_single', action='store_true')

    @classmethod
    def from_args(cls, args, train_data):
        n_classes = train_data.corpus.n_classes
        feature_dim = train_data.feature_dim

        allow_self_transitions = True

        assert args.sm_max_span_length is not None
        if args.sm_constrain_transitions:
            # assert args.task_specific_steps, "will get bad results with --sm_constrain_transitions if you don't also pass --task_specific_steps, because of multiple exits"
            # TODO: figure out what I meant by multiple exits; this seems fine at least if you're using the ComponentSemiMarkovModule. maybe add a check for this?
            # if not args.remove_background:
            #     raise NotImplementedError("--sm_constrain_transitions without --remove_background ")

            (
                allowed_starts, allowed_transitions, allowed_ends, ordered_indices_by_task
            ) = train_data.get_allowed_starts_and_transitions()
            if allow_self_transitions:
                for src in range(n_classes):
                    if src not in allowed_transitions:
                        allowed_transitions[src] = set()
                    allowed_transitions[src].add(src)
        else:
            allowed_starts, allowed_transitions, allowed_ends, ordered_indices_by_task = None, None, None, None

        if args.annotate_background_with_previous and not args.no_merge_classes:
            merge_classes = {}
            for task, indices in train_data.corpus._indices_by_task.items():
                background_indices = [ix for ix in indices if ix in train_data.corpus._background_indices]
                nonbackground_indices = [ix for ix in indices if ix not in train_data.corpus._background_indices]
                canon_bkg_ix = background_indices[0]
                for ix in background_indices:
                    if ix in merge_classes:
                        assert merge_classes[ix] == canon_bkg_ix
                    else:
                        merge_classes[ix] = canon_bkg_ix
                    # assert ix not in merge_classes
                for ix in nonbackground_indices:
                    if ix in merge_classes:
                        assert merge_classes[ix] == ix
                    else:
                        merge_classes[ix] =  ix
                    # assert ix not in merge_classes
                    # merge_classes[ix] = ix
        else:
            merge_classes = None

        if args.sm_component_model:
            if args.sm_component_decompose_steps:
                # assert not args.task_specific_steps, "can't decompose steps unless steps are across tasks; you should remove --task_specific_steps"
                n_components = train_data.corpus.n_components
                class_to_components = copy.copy(train_data.corpus.label_indices2component_indices)
            else:
                n_components = n_classes
                class_to_components = {
                    cls: {cls}
                    for cls in range(n_classes)
                }
            model = ComponentSemiMarkovModule(
                args,
                n_classes,
                n_components=n_components,
                class_to_components=class_to_components,
                feature_dim=feature_dim,
                allow_self_transitions=allow_self_transitions,
                allowed_starts=allowed_starts,
                allowed_transitions=allowed_transitions,
                allowed_ends=allowed_ends,
                merge_classes=merge_classes,
            )
        else:
            model = SemiMarkovModule(
                args,
                n_classes,
                feature_dim,
                allow_self_transitions=allow_self_transitions,
                allowed_starts=allowed_starts,
                allowed_transitions=allowed_transitions,
                allowed_ends=allowed_ends,
                merge_classes=merge_classes,
            )
        return SemiMarkovModel(args, n_classes, feature_dim, model, ordered_indices_by_task)

    def __init__(self, args, n_classes, feature_dim, model, ordered_indices_by_task=None):
        self.args = args
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.model = model
        self.ordered_indices_by_task = ordered_indices_by_task
        if args.cuda:
            self.model.cuda()

    def fit_supervised(self, train_data: Datasplit, task: int=None, task_indices: dict=None):
        assert not self.args.sm_component_model
        assert not self.args.sm_constrain_transitions
        loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=False, batch_size=1)
        features, labels = [], []
        for batch in loader:
            features.append(batch['features'].squeeze(0))
            labels.append(batch['gt_single'].squeeze(0))
        self.model.fit_supervised(features, labels, task=task, task_indices=task_indices)

    def make_additional_allowed_ends(self, tasks, lengths):
        if self.ordered_indices_by_task is not None:
            addl_allowed_ends = []
            for task, length in zip(tasks, lengths):
                ord_indices = self.ordered_indices_by_task[task]
                if length.item() < len(ord_indices):
                    this_allowed_ends = [ord_indices[length.item()-1]]
                else:
                    this_allowed_ends = []
                addl_allowed_ends.append(this_allowed_ends)
        else:
            addl_allowed_ends = None
        return addl_allowed_ends

    def expand_constraints(self, datasplit, task, task_indices, constraints):
        task_indices = list(task_indices.cpu().numpy())
        step_indices = datasplit.get_ordered_indices_no_background()[task]
        # constraints: batch_dim x T x K
        assert constraints.size(2) == len(step_indices)
        constraints_expanded = torch.zeros((constraints.size(0), constraints.size(1), len(task_indices)))
        for index, label in enumerate(step_indices):
            constraints_expanded[:,:,task_indices.index(label)] = constraints[:,:,index]
        return constraints_expanded

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None, task=None, task_indices=None):
        self.model.train()
        self.model.flatten_parameters()
        if use_labels:
            assert not self.args.sm_constrain_transitions
        initialize = True
        if use_labels and self.args.sm_supervised_method in ['closed-form', 'closed-then-gradient']:
            self.fit_supervised(train_data, task=task, task_indices=task_indices)
            if self.args.sm_supervised_method == 'closed-then-gradient':
                initialize = False
                callback_fn(-1, {})
            else:
                return
        if self.args.sm_init_non_projection_parameters_from:
            initialize = False
            if callback_fn:
                callback_fn(-1, {})
        optimizer, scheduler = make_optimizer(self.args, self.model.parameters())
        big_loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=True, batch_size=100)
        samp = next(iter(big_loader))
        big_features = samp['features']
        big_lengths = samp['lengths']
        if self.args.cuda:
            big_features = big_features.cuda()
            big_lengths = big_lengths.cuda()

        if initialize:
            self.model.initialize_gaussian(big_features, big_lengths)

        loader = make_data_loader(self.args, train_data, batch_by_task=True, shuffle=True, batch_size=self.args.batch_size)

        # print('{} videos in training data'.format(len(loader.dataset)))

        # all_features = [sample['features'] for batch in loader for sample in batch]
        # if self.args.cuda:
        #     all_features = [feats.cuda() for feats in all_features]

        C = self.n_classes
        K = self.args.sm_max_span_length

        for epoch in range(self.args.epochs):
            start_time = time.time()
            # call here since we may set eval in callback_fn
            self.model.train()
            losses = []
            multi_batch_losses = []
            nlls = []
            kls = []
            log_dets = []
            num_frames = 0
            num_videos = 0
            train_nll = 0
            train_kl = 0
            train_log_det = 0
            # for batch_ix, batch in enumerate(tqdm.tqdm(loader, ncols=80)):
            for batch_ix, batch in enumerate(loader):
                if self.args.train_limit and batch_ix >= self.args.train_limit:
                    break
                # if self.args.cuda:
                #     features = features.cuda()
                #     task_indices = task_indices.cuda()
                #     gt_single = gt_single.cuda()
                tasks = batch['task_name']
                videos = batch['video_name']
                features = batch['features']
                task_indices = batch['task_indices']
                lengths = batch['lengths']

                if 'train' in self.args.sm_constrain_with_narration:
                    assert all_equal(tasks)
                    constraints_expanded = self.expand_constraints(
                        train_data, tasks[0], task_indices[0], 1 - batch['constraints']
                    )
                    constraints_expanded *= self.args.sm_constrain_narration_weight
                else:
                    constraints_expanded = None

                num_frames += lengths.sum().item()
                num_videos += len(lengths)

                # assert len( task_indices) == self.n_classes, "remove_background and multi-task fit() not implemented"

                if self.args.cuda:
                    features = features.cuda()
                    lengths = lengths.cuda()
                    if constraints_expanded is not None:
                        constraints_expanded = constraints_expanded.cuda()

                if use_labels:
                    labels = batch['gt_single']
                    if self.args.cuda:
                        labels = labels.cuda()
                    spans = semimarkov_utils.labels_to_spans(labels, max_k=K)
                    use_mean_z = True
                else:
                    spans = None
                    use_mean_z = False

                addl_allowed_ends = self.make_additional_allowed_ends(tasks, lengths)

                ll, log_det = self.model.log_likelihood(features,
                                                 lengths,
                                                 valid_classes_per_instance=task_indices,
                                                 spans=spans,
                                                 add_eos=True,
                                                 use_mean_z=use_mean_z,
                                                 additional_allowed_ends_per_instance=addl_allowed_ends,
                                                 constraints=constraints_expanded,
                                                 tasks=tasks)
                nll = -ll
                kl = self.model.kl.mean()
                if use_labels:
                    this_loss = nll - log_det
                else:
                    this_loss = nll - log_det + kl
                multi_batch_losses.append(this_loss)
                nlls.append(nll.item())
                kls.append(kl.item())
                log_dets.append(log_det.item())

                train_nll += (nll.item() * len(videos))
                train_kl += (kl.item() * len(videos))
                train_log_det += (log_det.item() * len(videos))

                losses.append(this_loss.item())

                if len(multi_batch_losses) >= self.args.batch_accumulation:
                    loss = sum(multi_batch_losses) / len(multi_batch_losses)
                    loss.backward()
                    multi_batch_losses = []

                    if self.args.print_every and (batch_ix % self.args.print_every == 0):
                        param_norm = sum([p.norm()**2 for p in self.model.parameters()
                                          if p.requires_grad]).item()**0.5
                        gparam_norm = sum([p.grad.norm()**2 for p in self.model.parameters()
                                           if p.requires_grad and p.grad is not None]).item()**0.5
                        log_str = 'Epoch: %02d, Batch: %03d/%03d, |Param|: %.6f, |GParam|: %.2f, lr: %.2E, ' + \
                                  'loss: %.4f, recon: %.4f, kl: %.4f, log_det: %.4f, recon_bound: %.2f, Throughput: %.2f vid / sec'
                        print(log_str %
                              (epoch, batch_ix, len(loader), param_norm, gparam_norm,
                               optimizer.param_groups[0]["lr"],
                               (train_nll + train_kl + train_log_det) / num_videos, # loss
                               train_nll / num_frames, # recon
                               train_kl / num_frames, # kl
                               train_log_det / num_videos, # log_det
                               (train_nll + train_kl) / num_frames, # recon_bound
                              num_videos / (time.time() - start_time))) # Throughput
                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    self.model.zero_grad()
            train_loss = np.mean(losses)
            if scheduler is not None:
                scheduler.step(train_loss)
            callback_fn(epoch, {'train_loss': train_loss,
                                'train_nll_frame_avg': train_nll / num_frames,
                                'train_kl_vid_avg': train_kl / num_videos,
                                'train_recon_bound': (train_nll + train_kl) / num_frames})

    def predict(self, test_data):
        self.model.eval()
        self.model.flatten_parameters()
        predictions = {}
        loader = make_data_loader(self.args, test_data, shuffle=False, batch_by_task=True, batch_size=self.args.batch_size)
        # print('{} videos in prediction data'.format(len(loader.dataset)))
        # for batch in tqdm.tqdm(loader, ncols=80):
        for batch in loader:
            features = batch['features']
            task_indices = batch['task_indices']
            lengths = batch['lengths']

            # add a batch dimension
            # lengths = torch.LongTensor([features.size(0)]).unsqueeze(0)
            # features = features.unsqueeze(0)
            # task_indices = task_indices.unsqueeze(0)

            videos = batch['video_name']
            tasks = batch['task_name']
            assert len(set(tasks)) == 1
            task = next(iter(tasks))

            if 'test' in self.args.sm_constrain_with_narration:
                assert all_equal(tasks)
                constraints_expanded = self.expand_constraints(
                    test_data, task, task_indices[0], 1 - batch['constraints']
                )
                constraints_expanded *= self.args.sm_constrain_narration_weight
            else:
                constraints_expanded = None

            if self.args.cuda:
                features = features.cuda()
                task_indices = [ti.cuda() for ti in task_indices]
                lengths = lengths.cuda()
                if constraints_expanded is not None:
                    constraints_expanded = constraints_expanded.cuda()

            addl_allowed_ends = self.make_additional_allowed_ends(tasks, lengths)

            def predict(constraints):
                # TODO: figure out under which eval conditions use_mean_z should be False
                pred_spans = self.model.viterbi(features, lengths, task_indices, add_eos=True, use_mean_z=True,
                                                additional_allowed_ends_per_instance=addl_allowed_ends,
                                                constraints=constraints, corpus_index2label=test_data._corpus.index2label, task=task)
                pred_labels = semimarkov_utils.spans_to_labels(pred_spans)
                # if self.args.sm_predict_single:
                #     # pred_spans: batch_size x T
                #     pred_labels_single = torch.zeros_like(pred_labels)
                #     for i in pred_labels.size(0):
                #         for lab in torch.unique(pred_labels[i,:lengths[i]]):
                #             #emission_scores: b x N x C
                #             pred_labels
                #             pass

                # if self.args.sm_constrain_transitions:
                #     all_pred_span_indices = [
                #         [ix for ix, count in this_rle_spans]
                #         for this_rle_spans in semimarkov_utils.rle_spans(pred_spans, lengths)
                #     ]
                #     for i, indices in enumerate(all_pred_span_indices):
                #         remove_cons_dups = [ix for ix, group in itertools.groupby(indices)
                #                             if not ix in test_data.corpus._background_indices]
                #         non_bg_indices = [
                #             ix for ix in test_data.corpus.indices_by_task(task)
                #             if ix not in test_data.corpus._background_indices
                #         ]
                #         if len(remove_cons_dups) != len(non_bg_indices) and lengths[i].item() != len(remove_cons_dups):
                #             print("deduped: {}, indices: {}, length {}".format(
                #                 remove_cons_dups, non_bg_indices, lengths[i].item()
                #             ))
                #             # assert lengths[i].item() < len(non_bg_indices)

                pred_labels_trim_s = self.model.trim(pred_labels, lengths, check_eos=True)
                return pred_labels_trim_s

            pred_labels_trim_s = predict(constraints_expanded)

            # assert len(pred_labels_trim_s) == 1, "batch size should be 1"
            for ix, (video, pred_labels_trim) in enumerate(zip(videos, pred_labels_trim_s)):
                preds = pred_labels_trim.numpy()
                predictions[video] = preds
                # if constraints_expanded is not None:
                #     this_cons = batch['constraints'][ix]
                #     if this_cons.sum() > 0:
                #         step_indices = test_data.get_ordered_indices_no_background()[task]
                #         for t, label in enumerate(preds):
                #             if label in step_indices:
                #                 label_ix = step_indices.index(label)
                #                 assert batch['constraints'][ix,t,label_ix] == 1
                assert self.model.n_classes not in predictions[video], "predictions should not contain EOS: {}".format(
                    predictions[video])
        return predictions
