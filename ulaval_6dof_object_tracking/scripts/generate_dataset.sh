#!/usr/bin/env bash

SAVE_PATH=/media/ssd/deeptracking/test  # Path to save
MODEL_FILE=dragon.yml                   # Model file containing all the objects path to render
TRAIN_SAMPLES=200000                    # Number of training samples
VALID_SAMPLES=20000                     # Number of validation samples
MAX_TRANSLATION=0.02                    # Maximum translation (meter)
MAX_ROTATION=15                         # Maximum rotation (degree)
BOUNDING_BOX=0                          # Bounding box ratio w.r.t. maximum vertex distance in the model (10 => 110% and -10 => 90%)
RESOLUTION=174                          # Resolution of the image samples
DATA_TYPE=numpy                         # save type : numpy => large but fast to load. png => small put slower to load.

cd ..
export PYTHONPATH=$PYTHONPATH:"/home/mathieu/source/6DOF_tracking_evaluation"    # add your project path to PythonPath
echo Generating training data...
python3 dataset_generator.py -o ${SAVE_PATH}/train \
                             -m ${MODEL_FILE} \
                             -s ${TRAIN_SAMPLES} \
                             --show -t ${MAX_TRANSLATION} -r ${MAX_ROTATION} --boundingbox ${BOUNDING_BOX} \
                             -e ${RESOLUTION} --saveformat ${DATA_TYPE}

echo Generating validation data..
python3 dataset_generator.py -o ${SAVE_PATH}/train \
                             -m ${MODEL_FILE} \
                             -s ${VALID_SAMPLES} \
                             --show -t ${MAX_TRANSLATION} -r ${MAX_ROTATION} --boundingbox ${BOUNDING_BOX} \
                             -e ${RESOLUTION} --saveformat ${DATA_TYPE}

