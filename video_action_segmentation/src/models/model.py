import torch.optim
from torch.utils.data import DataLoader

from data.corpus import Datasplit


def add_training_args(parser):
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--workers', type=int, default=0)

    parser.add_argument('--max_grad_norm', type=float, default=10)

    parser.add_argument('--print_every', type=int, default=100)

    parser.add_argument('--no_reduce_plateau', action='store_true')
    parser.add_argument('--reduce_plateau_factor', type=float, default=0.2)
    parser.add_argument('--reduce_plateau_patience', type=float, default=1)
    parser.add_argument('--reduce_plateau_min_lr', type=float, default=1e-4)

    parser.add_argument('--train_limit', type=int)

    parser.add_argument('--dev_decode_frequency', type=int, default=1)


def make_optimizer(args, parameters):
    opt = torch.optim.Adam(parameters, lr=args.lr)
    if not args.no_reduce_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=args.reduce_plateau_factor,
            verbose=True,
            patience=args.reduce_plateau_patience,
            min_lr=1e-4,
            threshold=1e-5,
        )
    else:
        scheduler = None
    return opt, scheduler


def padding_colate(data_samples):
    data_samples = [samp for samp in data_samples if samp is not None]
    unpacked = {
        key: [samp[key] for samp in data_samples]
        for key in next(iter(data_samples)).keys()
    }

    lengths = [feats.size(0) for feats in unpacked['features']]
    # batch_size = len(lengths)
    # max_length = max(lengths)
    # lengths_t = torch.LongTensor(lengths)

    pad_keys = ['gt_single', 'features', 'constraints']
    nopad_keys = ['task_name', 'video_name', 'task_indices', 'gt', 'gt_with_background']
    data = {k: v for k, v in unpacked.items() if k in nopad_keys}
    data['lengths'] = torch.LongTensor(lengths)

    for key in pad_keys:
        if key in unpacked:
            data[key] = torch.nn.utils.rnn.pad_sequence(unpacked[key], batch_first=True, padding_value=0)

    return data


def make_data_loader(args, datasplit: Datasplit, shuffle, batch_by_task, batch_size=1):
    # assert batch_size == 1, "other sizes not implemented"
    return DataLoader(
        datasplit,
        # batch_size=batch_size,
        num_workers=args.workers,
        # shuffle=shuffle,
        # drop_last=False,
        # collate_fn=lambda batch: batch,
        collate_fn=padding_colate,
        batch_sampler=datasplit.batch_sampler(batch_size, batch_by_task, shuffle)
    )


class Model(object):
    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        raise NotImplementedError()

    def predict(self, test_data):
        raise NotImplementedError()
