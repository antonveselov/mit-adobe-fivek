import torch
import numpy as np


class ToTensor:
    def __call__(self, sample):
        sample['x'] = torch.from_numpy(np.transpose(sample['x'].astype(np.float32), (2, 0, 1)))
        sample['y'] = torch.from_numpy(np.transpose(sample['y'].astype(np.float32), (2, 0, 1)))
        return sample


class RandomCrop:
    def __init__(self, crop_size=128):
        self.crop_size = crop_size

    def __call__(self, sample):
        x = sample['x']
        y = sample['y']

        _, h, w = x.shape
        crop_x = np.random.randint(0, (w - self.crop_size) // 2) * 2
        crop_y = np.random.randint(0, (h - self.crop_size) // 2) * 2

        x = x[:, crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]
        y = y[:, crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]

        return {'x': x,
                'y': y}
