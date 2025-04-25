"""
Code modified from: https://github.com/diffusion-classifier/diffusion-classifier/blob/e9f772d78b75976112d88a1002c496f6ef0cb27e/eval_prob_dit.py
Original comment:
Some of the helper functions are taken from the original DiT repository.
https://github.com/facebookresearch/DiT
"""
import argparse
import random
import numpy as np
import os
import os.path as osp
import fnmatch
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.models import AutoencoderKL
from diffusion.datasets import get_target_dataset
from diffusion.utils import LOG_DIR
from DiT.diffusion import create_diffusion
from DiT.download import find_model
from DiT.models import DiT_XL_2
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def get_transform(image_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def main():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='val', choices=['val'], help='Name of split')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')

    # noises and ts
    parser.add_argument('--n_loops', type=int, default=20, help='Number of loops over (n_times // t_interval)')
    parser.add_argument('--t_interval', type=int, default=4, help='Timestep interval')
    parser.add_argument('--shuffle_ts', action='store_true', help='Shuffle timesteps at each loop')
    parser.add_argument('--noise_path', type=str, default='noise_256.pt', help='Path to shared noise to use')
    parser.add_argument('--val_freq', type=int, default=1, help='Frequency of val loss (monitor) calculation')

    # optim
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size')
    parser.add_argument('--accumulate_batches', type=int, default=1, help='Number of batches to accumulate gradients over')
    parser.add_argument('--discrete_optim', action='store_true', help='In-vocabulary discrete optimization')
    parser.add_argument('--nn_metric', type=str, default='l2', choices=['cosine', 'l2'], help='Nearest neighbour metric, for discrete_optim')

    # misc
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--allow_tf32', action='store_true', help='Allow TF32')

    args = parser.parse_args()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Running with TF32 enabled")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("Running with TF32 disabled")

    print("Running in deterministic cudnn mode")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # dataset
    dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=get_transform(args.image_size))
    image_idxs = list(range(len(dataset)))
    exp_folder = f"{args.dataset}{args.image_size}_inversion"
    if args.subset_path is not None:
        subset = np.load(args.subset_path)
        image_idxs = list(subset)
        print(f'Loaded subset of {len(image_idxs)} images from file {args.subset_path}')
        subset_file_name = osp.basename(args.subset_path).split(".")[0]
        exp_folder += f"_{subset_file_name}"
    else:
        raise Warning(f"Not specified `subset_path`, using all images in {args.dataset} {args.split} split."
                      f"We use a subset of 2*1000 images of imagenet val set for evaluation.")

    # log folder
    exp_name = f"t{args.t_interval}"
    if args.shuffle_ts:
        exp_name += "shuffle"
    if args.discrete_optim:
        exp_name += "_discOpt"
    exp_name += "_cosMtrc" if args.nn_metric == "cosine" else "_l2Mtrc"
    exp_name += f"_bs{args.batch_size}accum{args.accumulate_batches}_lr{args.lr}_{args.n_loops}loops"
    exp_name += f"_valFreq{args.val_freq}"

    run_folder = osp.join(LOG_DIR, exp_folder, exp_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # load models
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    vae.train = disabled_train
    for param in vae.parameters():
        param.requires_grad = False

    image_size = args.image_size
    latent_size = int(image_size) // 8
    model = DiT_XL_2(input_size=latent_size,
                     discrete_optim=args.discrete_optim,
                     nn_metric=args.nn_metric).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False

    # get y embeddings for further comparison
    pretrained_embs = model.y_embedder.embedding_table.weight.data
    if pretrained_embs.shape[0] == (model.y_embedder.num_classes + 1):
        pretrained_embs = pretrained_embs[:-1]
    if args.nn_metric == "cosine":
        pretrained_embs_normed = pretrained_embs / pretrained_embs.norm(dim=-1, keepdim=True)


    # build diffusion and ts
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    n_train_timesteps = diffusion.num_timesteps
    assert n_train_timesteps % args.t_interval == 0

    print(f"Elapse time (loading model): {time.time() - start:.2f}s")

    for img_idx in image_idxs:
        # start eval a new image
        print(f"\n==== Evaluating image {img_idx} ====\n")
        file_folder = osp.join(run_folder, str(img_idx))
        os.makedirs(file_folder, exist_ok=True)

        # seed everything
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # build noises and ts
        if args.noise_path is not None:
            noises = torch.load(args.noise_path)
            assert noises.shape[1:] == (4, latent_size, latent_size)
            noises = noises[:n_train_timesteps // args.t_interval]
            noises = noises.to(device)
        else:
            noises = torch.randn((n_train_timesteps // args.t_interval, 4, latent_size, latent_size), device=device)
        ts = torch.tensor(range(args.t_interval // 2, n_train_timesteps, args.t_interval)).to(device)
        assert ts.shape[0] % (args.batch_size * args.accumulate_batches) == 0, "Num of used ts should be divisible by actual bsz"

        # noises and ts for the "val" loss monitor
        noises_val = noises.clone()
        ts_val = ts.clone()

        # build learnable embedding
        dim = model.y_embedder.embedding_table.embedding_dim
        # emb init
        mean = model.y_embedder.embedding_table.weight.mean().item()
        std = model.y_embedder.embedding_table.weight.std().item()
        params = torch.randn(1, dim) * std + mean
        learnable_emb = torch.nn.Parameter(params.to(device), requires_grad=True)
        opt = torch.optim.AdamW([learnable_emb], lr=args.lr)

        # load image & calculate latent
        image = dataset[img_idx][0]
        label_gt = dataset[img_idx][1]  # for comparing
        time_inner = _time_inner = time.time()
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215

        best_val_loss = np.inf
        # optimization loop: loop over ts
        for loop_idx in range(args.n_loops):
            if args.shuffle_ts:
                shuffled_idx = torch.randperm(ts.shape[0])
                ts = ts[shuffled_idx]
                noises = noises[shuffled_idx]

            for batch_idx in range(0, ts.shape[0], args.batch_size):
                if batch_idx == 0:
                    opt.zero_grad()
                batch_ts = ts[batch_idx: batch_idx + args.batch_size]
                batch_noises = noises[batch_idx: batch_idx + args.batch_size]
                batch_x0 = x0.repeat(args.batch_size, 1, 1, 1)

                # forward
                noised_x = diffusion.q_sample(batch_x0, batch_ts, batch_noises)
                model_output = model(noised_x, batch_ts, y=learnable_emb.repeat(args.batch_size, 1), learnable_y=True)
                B, C = noised_x.shape[:2]
                noise_pred, model_var_values = torch.split(model_output, C, dim=1)

                # loss
                loss = F.mse_loss(noise_pred, batch_noises)
                loss.backward()
                if (batch_idx // args.batch_size + 1) % args.accumulate_batches == 0:
                    opt.step()
                    opt.zero_grad()

            with torch.no_grad():
                print(f"Loop {loop_idx + 1}/{args.n_loops}\t"
                      f"Train ET {time.time() - time_inner:.2f}s\t| Total ET {time.time() - _time_inner:.2f}s\t")
                time_inner = time.time()

                # calculate the "val" loss (monitor)
                # the monitor will be used for selecting the ckpt, finally
                if (loop_idx + 1) % args.val_freq == 0:
                    val_losses = []
                    for batch_idx in range(0, ts_val.shape[0], args.batch_size):
                        batch_ts = ts_val[batch_idx: batch_idx + args.batch_size]
                        batch_noises = noises_val[batch_idx: batch_idx + args.batch_size]
                        batch_x0 = x0.repeat(args.batch_size, 1, 1, 1)
                        # forward
                        noised_x = diffusion.q_sample(batch_x0, batch_ts, batch_noises)
                        model_output = model(noised_x, batch_ts,
                                             y=learnable_emb.repeat(args.batch_size, 1), learnable_y=True)
                        B, C = noised_x.shape[:2]
                        noise_pred, model_var_values = torch.split(model_output, C, dim=1)
                        # loss
                        loss = F.mse_loss(noise_pred, batch_noises)
                        val_losses.append(loss.item())
                    val_losses_mean = np.mean(val_losses)

                    # check whether current learnable_emb ckpt is correct (i.e., compare with gt)
                    # this is only used for final reporting the accuracy
                    # we do not use this signal for terminating
                    #  1) find nearest emb of learnable_emb in the pretrained embs
                    if args.nn_metric == "cosine":
                        learnable_emb_normed = learnable_emb / learnable_emb.norm(dim=-1, keepdim=True)
                        similarity = torch.matmul(learnable_emb_normed, pretrained_embs_normed.T)
                        distance = 1 - similarity
                    elif args.nn_metric == "l2":
                        distance = torch.cdist(learnable_emb, pretrained_embs, p=2,
                                               compute_mode="donot_use_mm_for_euclid_dist")
                    else:
                        raise NotImplementedError
                    nearest_idx = torch.argmin(distance, dim=-1)
                    #  2) compare with gt
                    GT_eval_result = (nearest_idx.item() == label_gt)

                    # print and save
                    print(f"Loop {loop_idx + 1}/{args.n_loops}\t"
                          f"Val ET {time.time() - time_inner:.2f}s\t| Total ET {time.time() - _time_inner:.2f}s\t"
                          f"Loss: {val_losses_mean:.10f}\t"
                          f"{'GT-Compare: Correct' if GT_eval_result else 'GT-Compare: Wrong'}")
                    if val_losses_mean < best_val_loss:
                        best_val_loss = val_losses_mean
                        torch.save(learnable_emb, osp.join(file_folder,
                                                           f"loop={loop_idx + 1}-val_loss={val_losses_mean:.10f}"
                                                           f"{'-gtCorrect' if GT_eval_result else '-gtWrong'}"
                                                           f".pt"))
                    time_inner = time.time()


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print(f"\nElapsed time (loading model + total images): {time.time() - start:.2f}s")
