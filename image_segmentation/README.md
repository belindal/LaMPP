# LaMPP for Image Segmentation
This repository contains code for LaMPP of image segmentation.
We follow a simple generative model of image segmentation where the room labels generate the object segment labels, which generate the observations. We then perform joint inference over object labels to derive their optimal configuration given observation.
Large parts of the code for this task is adapted from [RedNet code](https://github.com/JindongJiang/RedNet).


## Preliminaries
To set up the repository, download packages listed in `requirements.txt`.


## Dataset and Model Checkpoints
We use the SUNRGB-D dataset to benchmark and train our model. You can download the data on the [official webpage](http://rgbd.cs.princeton.edu), unzip it, and place it with a folder tree like this,

```bash
DATAPATH # Some arbitrary path
├── SUNRGBD # The unzip folder of SUNRGBD.zip
└── SUNRGBDtoolbox # The unzip folder of SUNRGBDtoolbox.zip
```

You can download pre-trained checkpoint trained on the full SUNRGB-D dataset from [here](http://bit.ly/2KDLeu9).


## Usage
The language model priors used for LaMPP for image segmentation can be found in [`lm_priors_saved/gpt3_sunrgbd_obj_obj_similarity_binary.npy`](https://github.com/belindal/LaMPP/blob/main/image_segmentation/lm_priors_saved/gpt3_sunrgbd_obj_obj_similarity_binary.npy) and [`lm_priors_saved/gpt3_sunrgbd_obj_room_cooccurence_binary.npy`](https://github.com/belindal/LaMPP/blob/main/image_segmentation/lm_priors_saved/gpt3_sunrgbd_obj_room_cooccurence_binary.npy).
These probabilities were obtained by querying GPT3 `davinci-002`.

### Generating Priors
The script for generating these priors is located in the parent directory, under [`../get_lm_priors.py`](https://github.com/belindal/LaMPP/blob/main/get_lm_priors.py), and can be run using
```bash
export PYTHONPATH=..
# get object-object perceptual similarity priors
python ../get_lm_priors.py --query-config lm_query_configs/objobj_similarity_config.json --output-save-path <LM_PRIORS_SAVE_PATH>
# get object-room cooccurrence priors
python ../get_lm_priors.py --query-config lm_query_configs/objroom_cooccur_config.json --output-save-path <LM_PRIORS_SAVE_PATH>
```
where `LM_PRIORS_SAVE_PATH` is filepath where the logits get saved.

### Using Priors During Inference
To run LaMPP during inference, you can run the following command:
```bash
export PYTHONPATH=..
python RedNet_inference.py
    --last-ckpt <PATH_TO_TRAINED_REDNET_CKPT>
    -b <BATCH_SIZE>
    -o <OUTPUT_DIRECTORY>
    --data-dir <DATAPATH>
    --split [val|test]
    (--cuda)
    (--visualize)
```
The root path where you downloaded the data `DATAPATH` should be passed to the program using the `--data-dir <DATAPATH>` argument.
