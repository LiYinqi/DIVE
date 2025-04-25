# DIVE for Classification

## Preparation

### Environment
Create a conda environment with the following command:
```bash
conda env create -f dive_cls.yml
conda activate dive_cls
```

### Dataset
Download [ImageNet-1K](http://www.image-net.org/) validation set and place it at `datasets/imagenet/val`.

Note: 
Following [Diffusion Classifier](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf#page=6.46),
we evaluate on a subset of 2000 images (2 images per class).
The ids we use (first 2 for each class) are listed in [`datasets/imagenet_val_subset_2x1000_ids.npy`](datasets/imagenet_val_subset_2x1000_ids.npy).


## Run
1. Following [Diffusion Classifier](https://github.com/diffusion-classifier/diffusion-classifier), first save a set of noises that will be used for all images:
```bash
python save_noise.py --img_size 256
```

2. Then, run DIVE inversion:
```bash
python dive_inversion.py \
  --dataset="imagenet" --split="val" --noise_path="noise_256.pt" \
  --subset_path="./datasets/imagenet_val_subset_2x1000_ids.npy" \
  --t_interval=4 --shuffle_ts \
  --batch_size=25 --accumulate_batches=1 \
  --discrete_optim --nn_metric="l2" \
  --lr=1e-2 --n_loops=20 --val_freq=1 --allow_tf32
```
Note: We tested the above script on a single A100 with TF32 on (`--allow_tf32`). Results may be different if turning it off or using other GPUs. 

3. Finally, compute the accuracy from the output log folder of the above run:
```bash
python dive_get_acc.py --inv_exp_root_folder `inv_exp_root_folder` --subset_path "./datasets/imagenet_val_subset_2x1000_ids.npy"
``` 
where `inv_exp_root_folder` is the output log folder of the previous DIVE inversion run.
It should be `./logs/imagenet256_inversion_imagenet_val_subset_2x1000_ids/t4shuffle_discOpt_l2Mtrc_bs25accum1_lr0.01_20loops_valFreq1` if using the above default command of step 2.


## Acknowledgments

Code in this directory is developed with [Diffusion Classifier](https://github.com/diffusion-classifier/diffusion-classifier) and [DiT](https://github.com/facebookresearch/DiT).
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