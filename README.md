# LaMPP: Language Models as Probabilistic Priors for Perception and Action
LaMPP is a method for injecting priors derived from language models into probabilistic models of perception and action.
LaMPP is a generic method with specific instantiations for each task it's applied to.
Code is largely adapted from existing external task-specific models, with additional modifications on top to integrate priors from the language model.
For a more detailed discussion, [see the paper here.](https://arxiv.org/abs/2302.02801)

<img src="https://github.com/belindal/LaMPP/blob/main/imgs/teaser.png" width=530>

We separate the code for each task implemented in the paper (image segmentation, object navigation, and video-action segmentation) into its own individual directory.
See the individual `README.md` in each directory for instructions on how to run LaMPP for each task.


## Credit
To cite LaMPP, please use
```
@misc{https://doi.org/10.48550/arxiv.2302.02801,
  doi = {10.48550/ARXIV.2302.02801},
  url = {https://arxiv.org/abs/2302.02801},
  author = {Li, Belinda Z. and Chen, William and Sharma, Pratyusha and Andreas, Jacob},
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {LaMPP: Language Models as Probabilistic Priors for Perception and Action},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

Our implementations build off existing code for base models in each task:
* Image segmentation builds off of [RedNet](https://github.com/JindongJiang/RedNet).
* Object navigation builds off of the [Stubborn agent](https://github.com/Improbable-AI/Stubborn) and the [Habitat Challenge](https://github.com/facebookresearch/habitat-challenge).
* Video-action segmentation builds off of [this repository](https://github.com/dpfried/action-segmentation).

