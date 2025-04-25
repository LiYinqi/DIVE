import os
import glob
import argparse
import numpy as np


def get_acc(exp_root_folder, subset_path='./datasets/imagenet_val_subset_2x1000_ids.npy'):
    """
    Merge the results from different images and report the accuracy.
    :param exp_root_folder: the root folder of the inversion run
    :param subset_path: for sanity check
    """
    image_idxs = list(np.load(subset_path))
    num_of_inverted_imgs = len(image_idxs)

    found_img_ids = os.listdir(exp_root_folder)
    # remove non-folder items, maybe not needed
    found_img_ids = [item for item in found_img_ids if os.path.isdir(os.path.join(exp_root_folder, item))]

    # sanity check
    assert len(found_img_ids) == num_of_inverted_imgs
    assert sorted(image_idxs) == sorted([int(founds) for founds in found_img_ids])

    monitor = 'val_loss'
    best_ckpt_filename_format = 'loop=*-val_loss=*-gt*.pt'

    num_correct = 0

    for img_id_folder in found_img_ids:

        # get saved ckpts for this image id
        ckpt_filenames = glob.glob(os.path.join(exp_root_folder, img_id_folder, best_ckpt_filename_format))
        assert len(ckpt_filenames) >= 1

        # get the monitor val_loss values from the ckpt filenames
        monitor_values = [float(os.path.basename(ckpt_filename).split(monitor + '=')[-1].split('-')[0]) for ckpt_filename in ckpt_filenames]

        # find (locate) best idx, argmin
        best_idx = monitor_values.index(min(monitor_values))

        # judge whether it is correct
        num_correct += (os.path.basename(ckpt_filenames[best_idx]).split('-gt')[-1].split('.')[0] == "Correct")

    # print acc
    print(f'{num_correct}/{num_of_inverted_imgs} correct, acc={num_correct / num_of_inverted_imgs * 100:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy after the run of inversion processes over the images.")
    parser.add_argument('--inv_exp_root_folder', type=str, required=True, help="Root folder (contains image ids) of the inversion experiment.")
    parser.add_argument('--subset_path', type=str, default='./datasets/imagenet_val_subset_2x1000_ids.npy', help="Path to the image id subset file. For sanity check.")
    args = parser.parse_args()
    get_acc(args.inv_exp_root_folder, args.subset_path)