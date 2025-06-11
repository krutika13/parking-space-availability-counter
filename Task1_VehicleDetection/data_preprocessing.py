import os
import random
import shutil
import argparse

def create_training_subset(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, num_samples):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    images = os.listdir(src_img_dir)
    sampled = random.sample(images, num_samples)

    for img_file in sampled:
        base = os.path.splitext(img_file)[0]
        lbl_file = base + '.txt'

        img_path = os.path.join(src_img_dir, img_file)
        lbl_path = os.path.join(src_lbl_dir, lbl_file)

        if os.path.exists(lbl_path):
            shutil.copy(img_path, os.path.join(dst_img_dir, img_file))
            shutil.copy(lbl_path, os.path.join(dst_lbl_dir, lbl_file))
        else:
            print(f"Label missing for {img_file}, skipping...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset of training data.")
    parser.add_argument("--src_img_dir", type=str, default="train/images", help="Source image directory")
    parser.add_argument("--src_lbl_dir", type=str, default="train/labels", help="Source label directory")
    parser.add_argument("--dst_img_dir", type=str, default="train_subset/images", help="Destination image directory")
    parser.add_argument("--dst_lbl_dir", type=str, default="train_subset/labels", help="Destination label directory")
    parser.add_argument("--num_samples", type=int, default=1500, help="Number of samples to include")
    args = parser.parse_args()

    create_training_subset(
        args.src_img_dir,
        args.src_lbl_dir,
        args.dst_img_dir,
        args.dst_lbl_dir,
        args.num_samples
    )
