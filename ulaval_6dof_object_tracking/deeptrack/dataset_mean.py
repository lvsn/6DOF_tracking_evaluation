import argparse

from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from pytorch_toolbox.transformations.compose import Compose
from deep_6dof_tracking.data.data_augmentation import Occluder, HSVNoise, Background, GaussianNoise, \
    GaussianBlur, OffsetDepth, ToTensor, Transpose, DepthDownsample
import os
from tqdm import tqdm
import numpy as np
from torch.utils import data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute mean')
    parser.add_argument('-d', '--dataset', help="Dataset path", metavar="FILE")
    parser.add_argument('-b', '--background', help="Background path", metavar="FILE")
    parser.add_argument('-r', '--occluder', help="Occluder path", metavar="FILE")
    parser.add_argument('-n', '--ncore', help="number of cpu core to use, -1 is all core", action="store", default=-1, type=int)

    arguments = parser.parse_args()

    occluder_path = arguments.occluder
    background_path = arguments.background
    data_path = arguments.dataset
    data_path = os.path.expandvars(data_path)
    occluder_path = os.path.expandvars(occluder_path)
    background_path = os.path.expandvars(background_path)

    number_of_core = arguments.ncore
    if number_of_core == -1:
        number_of_core = os.cpu_count()
    batch_size = 64

    transformations_pre = [Compose([Occluder(occluder_path, 0.75)])]

    transformations_post = [Compose([HSVNoise(0.07, 0.05, 0.1),
                                     Background(background_path),
                                     GaussianNoise(2, 10),
                                     GaussianBlur(6),
                                     DepthDownsample(0.7),
                                     OffsetDepth(),
                                     Transpose(),
                                     ToTensor()])]

    train_dataset = DeepTrackLoader(os.path.join(data_path, "train"), transformations_pre, transformations_post)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=False,
                                   drop_last=True,
                                   )

    n = 20000
    channel_means = np.zeros(8)
    total = 0
    for i, (data, target) in tqdm(enumerate(train_loader), total=int(n / batch_size)):
        bufferA, bufferB = data
        bufferA_numpy = bufferA.cpu().numpy()
        bufferB_numpy = bufferB.cpu().numpy()
        buffer_numpy = np.concatenate((bufferA_numpy, bufferB_numpy), axis=1)
        channel_means += np.mean(buffer_numpy, axis=(0, 2, 3))
        total += 1
        if i * batch_size >= n:
            break
    channel_means = channel_means / total

    channel_std = np.zeros(8)
    total = 0
    for i, (data, target) in tqdm(enumerate(train_loader), total=int(n / batch_size)):
        bufferA, bufferB = data
        bufferA_numpy = bufferA.cpu().numpy()
        bufferB_numpy = bufferB.cpu().numpy()
        buffer_numpy = np.concatenate((bufferA_numpy, bufferB_numpy), axis=1)
        image_means = np.mean(buffer_numpy, axis=(0, 2, 3))
        channel_std += np.square(image_means - channel_means)
        total += 1
        if i * batch_size >= n:
            break
    channel_std = np.sqrt(channel_std / total)

    np.save(os.path.join(data_path, "mean.npy"), channel_means)
    np.save(os.path.join(data_path, "std.npy"), channel_std)
