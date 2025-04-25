# DIVE: Inverting Conditional Diffusion Models for Discriminative Tasks

**Authors:** Yinqi Li, Hong Chang, Ruibing Hou, Shiguang Shan, Xilin Chen

**Venue:** IEEE Transactions on Multimedia (Accepted), 2025

[![arXiv](https://img.shields.io/badge/arXiv-2504.17253-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2504.17253)

**Abstract:**
Diffusion models have shown remarkable progress in various generative tasks such as image and video generation. 
This paper studies the problem of leveraging pretrained diffusion models for performing discriminative tasks. 
Specifically, we extend the discriminative capability of pretrained frozen generative diffusion models from the classification task 
[[1]](https://arxiv.org/abs/2303.16203), [[2]](https://arxiv.org/abs/2303.15233) to the more complex object detection task, by “inverting” a pretrained layout-to-image diffusion model. 
To this end, a gradient-based discrete optimization approach for replacing the heavy prediction enumeration process, 
and a prior distribution model for making more accurate use of the Bayes’ rule, are proposed respectively. 
Empirical results show that this method is on par with basic discriminative object detection baselines on COCO dataset. 
In addition, our method can greatly speed up the previous diffusion-based method 
[[1]](https://arxiv.org/abs/2303.16203), [[2]](https://arxiv.org/abs/2303.15233) for classification without sacrificing accuracy.


## Usage

This project studies [Detection](./Detection) and [Classification](./Classification) tasks. 
Please enter into corresponding folders and follow the instructions in the README files. 


## Acknowledgments

This project is developed with several awesome repos:
[Latent Diffusion Models](https://github.com/CompVis/latent-diffusion) and [Textual Inversion](https://github.com/rinongal/textual_inversion) for detection, and
[Diffusion Classifier](https://github.com/diffusion-classifier/diffusion-classifier) and [DiT](https://github.com/facebookresearch/DiT) for classification.
We thank the authors for their great work and open-sourcing.


## Citation

If you find this code useful, please consider citing:

```bibtex
@article{li2025dive,
  title   = {{DIVE}: Inverting Conditional Diffusion Models for Discriminative Tasks},
  author  = {Yinqi Li and Hong Chang and Ruibing Hou and Shiguang Shan and Xilin Chen},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2504.17253}
}
```