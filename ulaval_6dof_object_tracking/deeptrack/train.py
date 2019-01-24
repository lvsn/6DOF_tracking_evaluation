import sys
import os
from multiprocessing import cpu_count
import argparse

from pytorch_toolbox.train_loop import TrainLoop
import torch
from torch import optim
from torch.utils import data

from pytorch_toolbox.io import yaml_dump
import numpy as np
from pytorch_toolbox.transformations.compose import Compose

from ulaval_6dof_object_tracking.deeptrack.data_augmentation import Occluder, HSVNoise, Background, GaussianNoise, \
    GaussianBlur, DepthDownsample, OffsetDepth, NormalizeChannels, ToTensor
from ulaval_6dof_object_tracking.deeptrack.deeptrack_callback import DeepTrackCallback
from ulaval_6dof_object_tracking.deeptrack.deeptrack_loader import DeepTrackLoader
from ulaval_6dof_object_tracking.deeptrack.deeptrack_net import DeepTrackNet

if __name__ == '__main__':
    #
    #   load configurations from
    #

    parser = argparse.ArgumentParser(description='Train DeepTrack')
    parser.add_argument('-o', '--output', help="Output path", metavar="FILE")
    parser.add_argument('-d', '--dataset', help="Dataset path", metavar="FILE")
    parser.add_argument('-b', '--background', help="Background path", metavar="FILE")
    parser.add_argument('-r', '--occluder', help="Occluder path", metavar="FILE")
    parser.add_argument('-f', '--finetune', help="finetune path", default="None")
    parser.add_argument('-c', '--from_last', help="Continue training from last checkpoint", action="store_true")

    parser.add_argument('-i', '--device', help="Gpu id", action="store", default=0, type=int)
    parser.add_argument('-w', '--weightdecay', help="weight decay", action="store", default=0.000001, type=float)
    parser.add_argument('-l', '--learningrate', help="learning rate", action="store", default=0.001, type=float)
    parser.add_argument('-k', '--backend', help="backend : cuda | cpu", action="store", default="cuda")
    parser.add_argument('-e', '--epoch', help="number of epoch", action="store", default=25, type=int)
    parser.add_argument('-s', '--batchsize', help="Size of minibatch", action="store", default=128, type=int)
    parser.add_argument('-m', '--sharememory', help="Activate share memory", action="store_true")
    parser.add_argument('-n', '--ncore', help="number of cpu core to use, -1 is all core", action="store", default=-1, type=int)
    parser.add_argument('-g', '--gradientclip', help="Activate gradient clip", action="store_true")
    parser.add_argument('--tensorboard', help="Size of minibatch", action="store_true")


    arguments = parser.parse_args()

    learning_rate = arguments.learningrate
    weight_decay = arguments.weightdecay
    device_id = arguments.device
    backend = arguments.backend
    epochs = arguments.epoch
    batch_size = arguments.batchsize
    use_shared_memory = arguments.sharememory
    number_of_core = arguments.ncore
    gradient_clip = arguments.gradientclip
    start_from_last = arguments.from_last
    use_tensorboard = arguments.tensorboard

    output_path = arguments.output
    occluder_path = arguments.occluder
    background_path = arguments.background
    data_path = arguments.dataset
    finetune_path = arguments.finetune

    #
    #   Load configurations from file
    #

    data_path = os.path.expandvars(data_path)
    output_path = os.path.expandvars(output_path)
    occluder_path = os.path.expandvars(occluder_path)
    background_path = os.path.expandvars(background_path)
    finetune_path = os.path.expandvars(finetune_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if number_of_core == -1:
        number_of_core = cpu_count()
    if backend == "cuda":
        torch.cuda.set_device(device_id)
    tensorboard_path = ""
    if use_tensorboard:
        tensorboard_path = os.path.join(output_path, "tensorboard_logs")

    #
    #   Instantiate models/loaders/etc.
    #
    loader_param = {}

    model_class = DeepTrackNet
    callbacks = DeepTrackCallback(output_path, is_dof_only=True)
    loader_class = DeepTrackLoader


    # Here we use the following transformations:
    images_mean = np.load(os.path.join(data_path, "mean.npy"))
    images_std = np.load(os.path.join(data_path, "std.npy"))
    # transfformations are a series of transform to pass to the input data. Here we have to build a list of
    # transforms for each inputs to the network's forward call

    pretransforms = [Compose([Occluder(occluder_path, 0.75)])]

    posttransforms = [Compose([HSVNoise(0.07, 0.05, 0.1),
                               Background(background_path),
                               GaussianNoise(2, 10),
                               GaussianBlur(6),
                               DepthDownsample(0.7),
                               OffsetDepth(),
                               NormalizeChannels(images_mean, images_std),
                               ToTensor()])]

    print("Load datasets from {}".format(data_path))
    train_dataset = loader_class(os.path.join(data_path, "train"), pretransforms, posttransforms, **loader_param)
    valid_dataset = loader_class(os.path.join(data_path, "valid"), pretransforms, posttransforms, **loader_param)

    # Save important information to output:
    print("Save meta data in {}".format(output_path))
    np.save(os.path.join(output_path, "mean.npy"), images_mean)
    np.save(os.path.join(output_path, "std.npy"), images_std)
    yaml_dump(os.path.join(output_path, "meta.yml"), train_dataset.metadata)

    # Instantiate the data loader needed for the train loop. These use dataset object to build random minibatch
    # on multiple cpu core
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=use_shared_memory,
                                   drop_last=True,
                                   )

    val_loader = data.DataLoader(valid_dataset,
                                 batch_size=batch_size,
                                 num_workers=number_of_core,
                                 pin_memory=use_shared_memory,
                                 )

    # Setup model
    model = model_class(image_size=int(train_dataset.metadata["image_size"]))
    if finetune_path != "None":
        finetune_path = os.path.expandvars(finetune_path)
        print("Finetuning path : {}".format(finetune_path))
        checkpoint = torch.load(finetune_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Instantiate the train loop and train the model.
    train_loop_handler = TrainLoop(model, train_loader, val_loader, optimizer, backend, gradient_clip,
                                   use_tensorboard=use_tensorboard, tensorboard_log_path=tensorboard_path)
    train_loop_handler.add_callback(callbacks)
    print("Training Begins:")
    train_loop_handler.loop(epochs, output_path, load_best_checkpoint=start_from_last, save_all_checkpoints=False)
    print("Training Complete")
