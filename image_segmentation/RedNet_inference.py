import argparse
import torch
import imageio
import skimage.transform
import torchvision
import os
import pathlib
import RedNet_model
import RedNet_data
from image_segmentation.utils.utils import (
    color_label, load_ckpt, label_colours,
    sunrgb_items as items,
    eval_metrics,
    ConfusionMatrix, rednet_rooms,
    image_h, image_w
)
from gpt3_utils import (
    gpt3_constrained_generation,
)
import numpy as np
from tqdm import tqdm
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
parser.add_argument('-r', '--rgb', default=None, metavar='DIR',
                    help='path to image')
parser.add_argument('-d', '--depth', default=None, metavar='DIR',
                    help='path to depth')
parser.add_argument('-o', '--output', default=None, metavar='DIR',
                    help='path to output')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to data directory')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-classes', default=37, type=int,
                    help='# of object classes')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='batch size for inference')
parser.add_argument('--debug', action='store_true', default=False,
                    help='run in debug mode')
parser.add_argument('--use-priors', default=False, action='store_true',
                    help='use language model (or gt) priors')
parser.add_argument('--roomobj-priors', type=str, default=(pathlib.Path(__file__).parent.resolve() / 'lm_priors_saved/gpt3_sunrgbd_obj_room_cooccurence_binary.npy'),
                    help='filepath to room-object cooccurrence priors')
parser.add_argument('--objobj-priors', type=str, default=(pathlib.Path(__file__).parent.resolve() / 'lm_priors_saved/gpt3_sunrgbd_obj_obj_similarity_binary.npy'),
                    help='filepath to object-object (perceptual) similarity priors')
parser.add_argument('--use-model-chaining', default=False, action='store_true',
                    help='use model chaining')
parser.add_argument('--gpt3-cache-fp', type=str, default=(pathlib.Path(__file__).parent.parent.resolve() / "gpt3_cache"),
                    help='Filepath to store a (local) cache of GPT3 query results')
parser.add_argument('--eval-on-subset', default=None, metavar='DIR',
                    help='path to file listing filepaths to subset of testing data we will evaluate on')
parser.add_argument('--split', type=str, default="val", choices=["train", "val", "test"],
                    help='which split of the data to evaluate on')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='save images')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
margin_size = 15
if args.use_model_chaining:
    engine = "text-davinci-002"
    gpt_scoring_cache_fn = args.gpt3_cache_fp / f"{engine}.jsonl"
    gpt3_scoring_cache = load_gpt3_cache(gpt_scoring_cache_fn)
img_logits_softmax_temp = 9


print("Loading GPT3 priors")
# p(A(n) r has o)
item2room_cooccurs = np.load(args.roomobj_priors, allow_pickle=True)
item2room_cooccurs = torch.tensor(item2room_cooccurs).to(device)
# p(The o1 looks like the o2)
item2item_similarity = np.load(args.objobj_priors, allow_pickle=True)
item2item_similarity = torch.tensor(item2item_similarity).to(device)
# add 100% along diagonal (100% of time object is/looks like itself)
item2item_similarity[torch.eye(item2item_similarity.shape[0]).to(device).bool()] = 1.0
# make symmetric
item2item_similarity /= item2item_similarity.sum(0).unsqueeze(-1)


model_chaining_lm_prefix = f"""The following are possible rooms you can be in: {', '.join(sorted(rednet_rooms))}.
===
You can see: bed, lamp, pillow, table.

You are in the bedroom.
The thing that looks like bed is actually bed.
The thing that looks like lamp is actually lamp.
The thing that looks like pillow is actually pillow.
The thing that looks like table is actually nightstand.

===
You can see: cabinet, curtain, chair, shower, towel.

You are in the bathroom.
The thing that looks like cabinet is actually cabinet.
The thing that looks like curtain is actually shower curtain.
The thing that looks like chair is actually toilet.
The thing that looks like shower is actually shower.
The thing that looks like towel is actually towel.
===
"""


def get_room_from_lmpriors(pred_probs):
    """
    infer likeliest room based on distribution over objects in image pixels
    """
    # (width, height, n_rooms)
    curr_room_probs = pred_probs.to(torch.float64).permute(1,2,0).matmul(item2room_cooccurs).log()
    curr_room_probs -= item2room_cooccurs.sum(0).log()
    curr_room_logprobs = curr_room_probs[pred_probs.sum(0) > 0].mean(0)
    curr_room_probs = F.softmax(curr_room_logprobs*10 + item2room_cooccurs.sum(0).log(), dim=0)
    return curr_room_probs, curr_room_logprobs


def model_chaining_relabel(pred_logits, pred_probs, gt_img_labels, orig_img, items):
    """
    Relabel each image segment according using model chaining
    """
    uncalib_pred_labels = pred_probs.argmax(0)
    log_pred_probs = pred_probs.log()
    pred_items = uncalib_pred_labels.unique()
    pred_items_filtered = pred_items[(pred_items != 0) & (pred_items != 1) & (pred_items != 21) & (pred_items != items.index("person"))]
    pred_items_names = [items[item] for item in pred_items_filtered]
    pred_items_names = sorted(pred_items_names)
    new_pred_probs = pred_probs.clone()
    new_pred_labels = uncalib_pred_labels.clone()
    prompt_so_far = f"{model_chaining_lm_prefix}You can see: {', '.join([item.replace('_', ' ') for item in pred_items_names])}.\n\nYou are in the "
    pred_room_idx = gpt3_constrained_generation(
        engine=engine,
        input_prefix=prompt_so_far,
        classes=rednet_rooms,
        cache=gpt3_scoring_cache,
        gpt3_file=gpt_scoring_cache_fn,
    )
    prompt_so_far += rednet_rooms[pred_room_idx]
    for item_name in pred_items_names:
        possible_items = [items[item_idx] for item_idx in range(len(items)) if item_idx not in [0,1,21,items.index("person")]]
        pred_item_idx = gpt3_constrained_generation(
            engine=engine,
            input_prefix=f"{prompt_so_far}.\nThe thing that looks likes {item_name} is actually ",
            classes=[item.replace('_', ' ') for item in possible_items],
            cache=gpt3_scoring_cache,
            gpt3_file=gpt_scoring_cache_fn,
        )
        old_label = items.index(item_name)
        new_label = items.index(possible_items[pred_item_idx])
        if old_label != new_label:
            new_pred_labels[uncalib_pred_labels == old_label] = new_label
            orig_pred_probs = new_pred_probs[old_label, uncalib_pred_labels == old_label].clone()
            new_pred_probs[old_label, uncalib_pred_labels == old_label] = new_pred_probs[new_label, uncalib_pred_labels == old_label]
            new_pred_probs[new_label, uncalib_pred_labels == old_label] = orig_pred_probs
    return new_pred_labels, new_pred_probs


def put_prior_over_labels(pred_logits, pred_probs, gt_img_labels, orig_img, items):
    """
    Relabel each image segment according to LaMPP priors
    """
    uncalib_pred_labels = pred_probs.argmax(0)
    log_pred_probs = pred_probs.log()
    pred_items = uncalib_pred_labels.unique()
    pred_probs_2 = F.softmax(pred_logits)
    all_chunk_cooccurrences = torch.zeros(log_pred_probs.shape).to(device)
    # get confidence of each item...
    pred_items_filtered = pred_items[(pred_items != 0) & (pred_items != 1) & (pred_items != 21) & (pred_items != items.index("person"))]
    curr_room_probs, curr_room_logprobs = get_room_from_lmpriors(pred_probs_2)
    cooccurrence_weights = (item2room_cooccurs * curr_room_probs).sum(-1)
    all_obj_positions_filtered = [(uncalib_pred_labels == i).nonzero().float().mean(0) for i in pred_items_filtered]
    all_obj_positions_filtered = torch.stack(all_obj_positions_filtered)
    for i in pred_items:
        obj_position = (uncalib_pred_labels == i).nonzero().float().mean(0)
        # log P(l_n|detected identity) 
        #  P(l_n|detected identity) = p(l_n, l_d) / sum_{l_n} p(l_n, l_d)
        identity_weights = item2item_similarity[i]
        all_chunk_cooccurrences[:,uncalib_pred_labels == i] += cooccurrence_weights.log().unsqueeze(-1) + identity_weights.log().unsqueeze(-1)

    iou_prev_labels = eval_metrics(log_pred_probs, gt_img_labels, unk_idx=-1)["perlabel_IoU_raw_counts"]
    log_pred_probs_new = F.softmax(pred_logits / img_logits_softmax_temp).log() + F.softmax(all_chunk_cooccurrences, dim=0).log()
    model_labels = log_pred_probs_new.argmax(0)
    iou_new_labels = eval_metrics(log_pred_probs_new, gt_img_labels, unk_idx=-1)["perlabel_IoU_raw_counts"]
    # renormalize
    pred_probs = F.softmax(log_pred_probs_new, dim=0)
    return model_labels, pred_probs, {
        "room_distr_logp": curr_room_logprobs.tolist(),
        "iou_pre_prior": iou_prev_labels, "iou_post_prior": iou_new_labels}


def inference():

    model = RedNet_model.RedNet(pretrained=False, num_classes=args.num_classes)
    load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    all_metrics = ConfusionMatrix(args.num_classes)
    all_metrics.reset()
    if args.data_dir:
        val_ds = RedNet_data.SUNRGBD(
            phase=args.split, data_dir=args.data_dir, val_subset_file=args.eval_on_subset, debug_mode=args.debug)
        if args.output or args.visualize:
            os.makedirs(args.output, exist_ok=True)
            if args.visualize:
                img_output_dir = os.path.join(os.path.abspath(args.output), "images")   
            if args.output:
                saved_labels_file = open(os.path.join(args.output, "results.jsonl"), "a")
    else:
        image = imageio.imread(args.rgb)
        depth = imageio.imread(args.depth)
        val_ds = [{"image": image, "depth": depth, "out": args.output}]
    avg_accuracy = 0
    avg_pixelwise_prob4goldlabel = 0
    avg_mIoU = 0
    avg_perlabel_IoU = {label: [] for label in items}
    perlabel_avgprob = {label: [] for label in items}
    all_pixel_labels = {label: [] for label in items}
    all_pixel_probs = {label: [] for label in items}

    pbar = tqdm(enumerate(val_ds), total=len(val_ds))

    gtobj_to_predlabel = {}
    gtobj_to_predcooccurs = {}
    predlabel_to_gtobj = {}
    with torch.no_grad():
        for ex_idx, ex in pbar:
            image = ex['image']
            depth = ex['depth']
            if image.dtype == np.uint8:
                image = image.astype(np.int)
                depth = depth.astype(np.int)
            # Bi-linear
            image_ds = skimage.transform.resize(image, (image_h, image_w), order=1,
                                                mode='reflect', preserve_range=True)
            # Nearest-neighbor
            depth_ds = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                                mode='reflect', preserve_range=True)
            if 'label' in ex:
                label = ex['label']
                label_ds = skimage.transform.resize(label, (image_h, image_w), order=0,
                                                    mode='reflect', preserve_range=True)
                label_ds -= 1

            if 'out' not in ex and args.visualize:
                data_dir = os.path.abspath(args.data_dir)
                ex['id'] = os.path.abspath(ex['id'])
                ex['out'] = ex['id'].replace(data_dir, img_output_dir).replace(".npy", ".png")
                assert ex['out'] != ex['id']
                os.makedirs(os.path.split(ex['out'])[0], exist_ok=True)

            image = image_ds / 255
            image = torch.from_numpy(image).float()
            depth = torch.from_numpy(depth_ds).float()
            image = image.permute(2, 0, 1)
            depth.unsqueeze_(0)

            image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])(image)
            depth = torchvision.transforms.Normalize(mean=[19050],
                                                    std=[9650])(depth)

            image = image.to(device).unsqueeze_(0)
            depth = depth.to(device).unsqueeze_(0)

            pred = model(image, depth)
            pred_probs = F.softmax(pred, 1).squeeze(0)
            pre_prior_pred_probs = pred_probs.clone()
            pred_labels = pred.argmax(1).squeeze(0)

            pre_prior_pred_labels = pred_labels.clone()

            # add priors
            if args.use_priors or args.use_model_chaining:
                # record actual label of NS
                for label in np.unique(label_ds).astype(np.int):
                    if label == -1: continue
                    if items[label] not in gtobj_to_predlabel:
                        gtobj_to_predlabel[items[label]] = {}
                    gtobj_pred_labels = pre_prior_pred_labels[label_ds == label]
                    for pred_label in gtobj_pred_labels.unique():
                        if items[pred_label] not in gtobj_to_predlabel[items[label]]:
                            gtobj_to_predlabel[items[label]][items[pred_label]] = 0
                        gtobj_to_predlabel[items[label]][items[pred_label]] += (gtobj_pred_labels == pred_label).sum().item()
                for pred_label in pre_prior_pred_labels.unique():
                    if items[pred_label]  not in predlabel_to_gtobj:
                        predlabel_to_gtobj[items[pred_label]] = {}
                    gt_labels = label_ds[(pre_prior_pred_labels == pred_label).cpu().numpy()]
                    gt_labels = gt_labels[gt_labels != -1].astype(np.int)
                    for gt_label in np.unique(gt_labels):
                        if items[gt_label] not in predlabel_to_gtobj[items[pred_label]]:
                            predlabel_to_gtobj[items[pred_label]][items[gt_label]] = 0
                        predlabel_to_gtobj[items[pred_label]][items[gt_label]] += (label_ds == gt_label).sum().item()

                    try:
                        gtobj = np.bincount(gt_labels).argmax()
                    except:
                        continue
                    if items[gtobj] not in gtobj_to_predcooccurs:
                        gtobj_to_predcooccurs[items[gtobj]] = {}
                    surrounding_labels = pre_prior_pred_labels.unique()
                    surrounding_labels = surrounding_labels[surrounding_labels != pred_label]
                    for surrounding_label in surrounding_labels:
                        if items[surrounding_label] not in gtobj_to_predcooccurs[items[gtobj]]:
                            gtobj_to_predcooccurs[items[gtobj]][items[surrounding_label]] = 0
                        gtobj_to_predcooccurs[items[gtobj]][items[surrounding_label]] += 1
                if args.use_model_chaining:
                    pred_labels, pred_probs = model_chaining_relabel(pred[0], pred_probs, gt_img_labels=label_ds, orig_img=image_ds, items=items)
                    room_distr = {}
                else:
                    pred_labels, pred_probs, room_distr = put_prior_over_labels(pred[0], pred_probs, gt_img_labels=label_ds, orig_img=image_ds, items=items)
                result_entries = {**room_distr}
            else:
                pred_labels = torch.max(pred, 1)[1][0]
                result_entries = {}

            # evaluation
            if 'label' in ex:
                all_metrics.update((pred_probs.unsqueeze(0), torch.tensor(label_ds).unsqueeze(0).to(device, torch.int64)))
                metrics = eval_metrics(pred_probs, label_ds, -1)
                avg_prob = all_metrics.avg_prob()
                avg_prob = avg_prob[avg_prob > 0].mean()
                mIoU = all_metrics.miou()
                mIoU = mIoU[mIoU > 0].mean()

                if args.output:
                    saved_labels_file.write(json.dumps({ex['out']: result_entries})+"\n")
                pbar.set_description(f"pixel accuracy: {all_metrics.accuracy()*100:.2f}, prob: {avg_prob*100:.2f}, mIoU: {mIoU*100:.2f}")

            # visualization
            if args.visualize and not os.path.exists(ex['out']):
                image = image[0]
                canvas = np.full([image.shape[1] * 2 + 3 * margin_size, image.shape[2] * 2 + 3 * margin_size, 3], 255)
                # draw real image
                canvas[margin_size:image.shape[1]+margin_size, margin_size:image.shape[2]+margin_size, :] = image_ds
                # draw labels
                if 'label' in ex:
                    labels = color_label(torch.from_numpy(label_ds), label_colours)[0]
                    canvas[
                        margin_size:image.shape[1]+margin_size,
                        image.shape[2]+2*margin_size:image.shape[2]+2*margin_size+labels.shape[2],
                        :,
                    ] = labels.cpu().numpy().transpose((1, 2, 0))
                # draw pre-prior preds
                pre_prior_output = color_label(pre_prior_pred_labels, label_colours)[0]
                canvas[
                    image.shape[1]+2*margin_size:image.shape[1]+2*margin_size+pre_prior_output.shape[1],
                    margin_size:pre_prior_output.shape[2]+margin_size,
                    :,
                ] =  pre_prior_output.cpu().numpy().transpose((1, 2, 0))
                # draw post-prior preds
                output = color_label(pred_labels, label_colours)[0]
                canvas[
                    image.shape[1]+2*margin_size:image.shape[1]+2*margin_size+output.shape[1],
                    image.shape[2]+2*margin_size:image.shape[2]+2*margin_size+output.shape[2],
                    :,
                ] =  output.cpu().numpy().transpose((1, 2, 0))

                im = plt.imshow(canvas, interpolation='none')
                # create a patch (proxy artist) for every color 
                classes_present_idx = set(np.unique(label_ds.astype(np.int))).union(set(np.unique(pred_labels.cpu().numpy()))).union(set(np.unique(pre_prior_pred_labels.cpu().numpy())))
                patches = [ mpatches.Patch(
                    color=[c/255 for c in label_colours[i]], label=items[i]
                ) for i in classes_present_idx if i > 0 ]
                # put those patched as legend-handles into the legend
                lgd = plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
                plt.savefig(ex['out'],  bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.clf()

    """
    Print results
    """
    print(f"Avg pixelwise accuracy: {avg_accuracy / len(val_ds)} {all_metrics.accuracy()}")
    print(f"Avg pixelwise gold label probs: {all_metrics.avg_prob().mean()}")
    print(f"Avg per-label IoU: {all_metrics.miou()}")
    for label in range(len(all_metrics.iou())):
        avg_perlabel_IoU[items[label]] = all_metrics.iou()[label].item()
        if len(perlabel_avgprob[items[label]]) > 0:
            print(f"    {items[label]}: {avg_perlabel_IoU[items[label]]} {sum(perlabel_avgprob[items[label]]) / len(perlabel_avgprob[items[label]])*100}")
        else:
            print(f"    {items[label]}: {avg_perlabel_IoU[items[label]]}")
        if items[label] in gtobj_to_predlabel:
            total_prob = sum(gtobj_to_predlabel[items[label]].values())
            gtobj_to_predlabel[items[label]] = {
                pred_label: gtobj_to_predlabel[items[label]][pred_label] / total_prob for pred_label in gtobj_to_predlabel[items[label]]
            }
            normalized_sorted_labels = {k: v for k,v in sorted(gtobj_to_predlabel[items[label]].items(), key=lambda item: item[1], reverse=True)[:5]}
            print("        GT->pred identity: " + str(normalized_sorted_labels))
        if items[label] in predlabel_to_gtobj:
            total_prob = sum(predlabel_to_gtobj[items[label]].values())
            predlabel_to_gtobj[items[label]] = {
                gt_label: predlabel_to_gtobj[items[label]][gt_label] / total_prob for gt_label in predlabel_to_gtobj[items[label]]
            }
            normalized_sorted_labels = {k: v for k,v in sorted(predlabel_to_gtobj[items[label]].items(), key=lambda item: item[1], reverse=True)[:5]}
            print("        pred->GT identity: " + str(normalized_sorted_labels))
        if items[label] in gtobj_to_predcooccurs:
            total_prob = sum(gtobj_to_predcooccurs[items[label]].values())
            gtobj_to_predcooccurs[items[label]] = {
                pred_label: gtobj_to_predcooccurs[items[label]][pred_label] / total_prob for pred_label in gtobj_to_predcooccurs[items[label]]
            }
            normalized_sorted_labels = {k: v for k,v in sorted(gtobj_to_predcooccurs[items[label]].items(), key=lambda item: item[1], reverse=True)[:5]}
            print("        cooccurs: " + str(normalized_sorted_labels))
    print(avg_perlabel_IoU)


if __name__ == '__main__':
    inference()
