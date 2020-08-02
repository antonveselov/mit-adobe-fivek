import sys
import os
import rawpy
import subprocess
import tqdm

import numpy as np
import pyexif

from glob import glob
from skimage import io
from skimage.measure import compare_ssim
from torch.utils.data import Dataset
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear


class FiveKDataset(Dataset):
    def __init__(self, fivek_dir="./fivek_dataset", part_id=0, partitions=(1000, 10, 10), transform=None):
        super(FiveKDataset, self).__init__()

        self.transform = transform

        self.inputs = []
        self.targets = []
        for input_ in sorted(glob(os.path.join(fivek_dir, "raw_photos/*.dng"))):
            fname = os.path.basename(input_)[:-4]
            fname_target = os.path.join(fivek_dir, f"expert_C/{fname}.tif")
            fname_input = os.path.join(fivek_dir, f"raw_photos/{fname}.dng")
            if os.path.exists(fname_input) and os.path.exists(fname_target):
                self.targets.append(fname_target)
                self.inputs.append(fname_input)

        part_start = 0
        for i in range(part_id):
            part_start += partitions[i]
        part_end = part_start + partitions[part_id]

        self.inputs = self.inputs[part_start:part_end]
        self.targets = self.targets[part_start:part_end]

        fname = "_".join([str(x) for x in partitions]) + "_" + str(part_id) + ".txt"
        if not os.path.exists(fname):
            self.inputs, self.targets, items = self.filter_out_hard_samples()
            with open(fname, "wt") as f:
                f.write(" ".join([str(x) for x in items]))
        else:
            items = open(fname, "rt").readlines()
            items = [int(x) for x in items[0].split(" ")]
            self.inputs = [self.inputs[x] for x in items]
            self.targets = [self.targets[x] for x in items]

    def filter_out_hard_samples(self):

        inputs = []
        targets = []
        items = []

        progress = tqdm.tqdm(range(len(self)), "Selecting samples from FiveK dataset", file=sys.stdout)
        for item in progress:
            h_input_ = rawpy.imread(self.inputs[item])
            input_ = h_input_.raw_image_visible
            target_ = io.imread(self.targets[item])

            ratio_input = input_.shape[0] / input_.shape[1]
            ratio_target = target_.shape[0] / target_.shape[1]

            # TODO: remove magic constant "0.05", use configurable threshold
            if np.abs(1 - ratio_input/ratio_target) < 0.05:
                inputs.append(self.inputs[item])
                targets.append(self.targets[item])
                items.append(item)
        progress.close()

        return inputs, targets, items

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        # flip raw image according to meta
        # TODO: use pyexif instead
        proc = subprocess.run(["exiftool", self.inputs[item]], check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)
        out = proc.stdout
        out = out.split("\n")

        crop = [0, 0]
        cfa_pattern_2 = None
        cfa_plane_color = None
        white_level = None
        black_level = 0
        for line in out:
            if "Default Crop Origin" in line:
                crop = [int(x) for x in line.split(" ")[-2:]]
            elif "CFA Pattern 2" in line:
                cfa_pattern_2 = [int(x) for x in line.split(" ")[-4:]]
            elif "CFA Plane Color" in line:
                cfa_plane_color = line.split(" ")[-1].split(",")
            elif "White Level" in line:
                white_level = int(line.split(" ")[-1])

        cfa_pattern = "".join([cfa_plane_color[x][0] for x in cfa_pattern_2])

        h_input_ = rawpy.imread(self.inputs[item])
        cfa = np.clip((h_input_.raw_image_visible - black_level) / white_level, 0, 1)  # account for impulse noise
        input_ = demosaicing_CFA_Bayer_bilinear(cfa, cfa_pattern)

        target_ = io.imread(self.targets[item]) / (2 ** 16 - 1)

        input_ = input_[crop[1]:crop[1] + target_.shape[0],
                        crop[0]:crop[0] + target_.shape[1], :]

        sample = {
            "x": input_,
            "y": target_
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    import torch
    import transforms

    from torchvision.transforms import Compose
    from torch.utils.data import DataLoader

    fivek_dataset = FiveKDataset(partitions=(1000, 10, 10),
                                 part_id=0,
                                 transform=Compose([transforms.ToTensor(),
                                                    transforms.RandomCrop(crop_size=512),
                                                    ]))

    print("Samples after cleaning:", len(fivek_dataset))

    train_loader = DataLoader(fivek_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress = tqdm.tqdm(train_loader, f"Data read progress", file=sys.stdout)
    losses = []
    for batch in progress:
        x = batch['x'].to(device=device)
        y_gt = batch['y'].to(device=device)

    progress.close()

    print("Done")
