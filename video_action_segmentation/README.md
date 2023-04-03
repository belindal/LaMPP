# LaMPP for Video Action Segmentation
![Generative Model for Video Action Segmentation](https://github.com/belindal/LaMPP/blob/main/imgs/actseg.png)

This repository contains code for LaMPP for action recognition and segmentation from video.
In this domain, we use LMs to place a prior over *model parameters*. The underlying model we use is an HSMM.
We follow a simple generative model of action-transition parameters where the task being demonstrated in the video generates a Dirichlet prior over  these transition probabilities.
We then perform joint inference (Viterbi algorithm) over action labels to derive their optimal configuration given observation.
Large parts of the code for this task is adapted from [this repository](https://github.com/dpfried/action-segmentation).

## Setup
### Environment
To set up the environment for running this model,
```bash
conda create -n actseg PYTHON=3.6
conda activate actseg
```

Next, install the following requirements:
* pytorch 1.3
* sklearn
* editdistance
* tqdm
* Particular commits of [genbmm](https://github.com/harvardnlp/genbmm) and [pytorch-struct](https://github.com/harvardnlp/pytorch-struct/). Newer versions may run out of memory on the long videos in the CrossTask dataset, due to changes to pytorch-struct that improve runtime complexity but increase memory usage. They can be installed via

```bash
pip install -U git+https://github.com/harvardnlp/genbmm@bd42837ae0037a66803218d374c78fda72a9c9f4
pip install -U git+https://github.com/harvardnlp/pytorch-struct@1c9b038a1bbece32fe8d2d46d9e3d7c09f4c08e7
```

See `env.yml` for a full list of other dependencies, which can be installed with conda.

### Dataset
1. Download and unpack the CrossTask dataset of Zhukov et al.:

```bash
cd data
mkdir crosstask
cd crosstask
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_constraints.zip
unzip '*.zip'
```

2. Preprocess the features with PCA. In the repository's root folder, run

```bash
PYTHONPATH="src/":$PYTHONPATH python src/data/crosstask.py
```

This should generate the folder `data/crosstask/crosstask_processed/crosstask_primary_pca-200_with-bkg_by-task`


## Usage
The language model priors used for LaMPP for video-action segmentation can be found in [`lm_priors_saved/gpt3_init_action_by_task.pkl.npy`](https://github.com/belindal/LaMPP/blob/main/video_action_segmentation/lm_priors_saved/gpt3_init_action_by_task.pkl.npy) and [`lm_priors_saved/gpt3_trans_action_by_task.pkl.npy`](https://github.com/belindal/LaMPP/blob/main/video_action_segmentation/lm_priors_saved/gpt3_trans_action_by_task.pkl.npy).
These probabilities were obtained by querying GPT3 `davinci-002`.

### Generating Priors
The script for generating these priors is located in the parent directory, under [`../get_lm_priors.py`](https://github.com/belindal/LaMPP/blob/main/get_lm_priors.py), and can be run using
```bash
export PYTHONPATH=..
# get prior p(initial action | task)
python ../get_lm_priors.py --query-config lm_query_configs/task_initaction_config.json --output-save-path <LM_PRIORS_SAVE_PATH>
# get prior p(next action | preceding action, task)
python ../get_lm_priors.py --query-config lm_query_configs/task_transaction_config.json --output-save-path <LM_PRIORS_SAVE_PATH>
```
where `LM_PRIORS_SAVE_PATH` is filepath where the logits get saved.

### Training Model
Train the HSMM on the full training set, skipping background scenes:
```bash
./run_crosstask_i3d-resnet-audio.sh pca_semimarkov_sup_nobkg \
    --classifier semimarkov \
    --training supervised \
    --remove_background \
    --cuda
```
Model weights will be saved to `expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg`.

**Training on OOD data**

To train on only a subset of videos, add `--train_subset <train_subset>`.

For example, in the paper we train on a subset of videos (located at [`data/crosstask/crosstask_release/videos_train_heldout_transition.csv`](https://github.com/belindal/LaMPP/blob/main/video_action_segmentation/data/crosstask/crosstask_release/videos_train_heldout_transition.csv)) created by holding out certain common action transitions and excluding any videos that demonstrate that action transition. To replicate this, use
```bash
./run_crosstask_i3d-resnet-audio.sh pca_semimarkov_sup_nobkg \
    --classifier semimarkov \
    --training supervised \
    --remove_background \
    --cuda \
    --train_subset train_heldout_transition
```

**Training Model with LM Priors over Parameters**

To use a GPT3 prior over HSMM (action transition) parameters, add `--use_lm_smoothing gpt3 --sm_supervised_state_smoothing <smoothing_param>`, whereby `<smoothing_param>` is a float specifying how much weight to put on the prior over the posterior (equivalently, the smoothing parameter) when learning the weights.

*Note*: The GPT3 priors must be located under:
* [`lm_priors_saved/gpt3_init_action_by_task.pkl.npy`](https://github.com/belindal/LaMPP/blob/main/video_action_segmentation/lm_priors_saved/gpt3_init_action_by_task.pkl.npy)
* [`lm_priors_saved/gpt3_trans_action_by_task.pkl.npy`](https://github.com/belindal/LaMPP/blob/main/video_action_segmentation/lm_priors_saved/gpt3_init_action_by_task.pkl.npy)

For example, in the paper we train on the heldout-transition subset with GPT3-based smoothing and smoothing parameter 10:
```bash
./run_crosstask_i3d-resnet-audio.sh pca_semimarkov_sup_nobkg_heldout_transition \
    --classifier semimarkov \
    --training supervised \
    --remove_background \
    --cuda \
    --train_subset train_heldout_transition \
    --use_lm_smoothing gpt3 \
    --sm_supervised_state_smoothing 10
```

### Evaluating Models
To evaluate trained models, run
```bash
./run_crosstask_i3d-resnet-audio.sh <save_path> \
    --classifier semimarkov \
    --training supervised \
    --remove_background \
    --cuda \
    --model_input_path expts/crosstask_i3d-resnet-audio/<save_path>/ \
    --prediction_output_path expts/crosstask_i3d-resnet-audio/<save_path>
```
whereby `<save_path>` is the saved model path (`pca_semimarkov_sup_nobkg` and `pca_semimarkov_sup_nobkg_heldout_transition` in the above two examples, respectively).

**Replacing Model Parameters with LM Priors during Evaluation**

*Note*: The GPT3 priors must be located under:
* [`lm_priors_saved/gpt3_init_action_by_task.pkl.npy`](https://github.com/belindal/LaMPP/blob/main/video_action_segmentation/lm_priors_saved/gpt3_init_action_by_task.pkl.npy)
* [`lm_priors_saved/gpt3_trans_action_by_task.pkl.npy`](https://github.com/belindal/LaMPP/blob/main/video_action_segmentation/lm_priors_saved/gpt3_init_action_by_task.pkl.npy)

To replace the HSMM action transition probabilities with GPT3 priors during evaluation, add `--saved_probabilities gpt3`:
```bash
./run_crosstask_i3d-resnet-audio.sh pca_semimarkov_sup_nobkg_gpt3_priors \
    --classifier semimarkov \
    --training supervised \
    --remove_background \
    --cuda \
    --model_input_path expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg/ \
    --prediction_output_path expts/crosstask_i3d-resnet-audio/pca_semimarkov_sup_nobkg_gpt3_priors \
    --saved_probabilities gpt3
```

