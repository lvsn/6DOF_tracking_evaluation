#!/usr/bin/env bash

DATA_PATH=/media/ssd/deeptracking/test
OUTPUT_PATH=/media/ssd/to_delete
BACKGROUND_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/SUN3D
OCCLUDER_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptrack_train/hand

export PYTHONPATH=$PYTHONPATH:"/home/mathieu/source/6DOF_tracking_evaluation"    # add your project path to PythonPath

echo Compute mean/std of dataset
python3 dataset_mean.py -d $DATA_PATH -b $BACKGROUND_PATH -r $OCCLUDER_PATH

echo Start training
python3 train.py \
        --output $OUTPUT_PATH/$1 \
        --dataset $DATA_PATH/$1 \
        --background $BACKGROUND_PATH \
        --occluder $OCCLUDER_PATH \
        --batchsize 12

