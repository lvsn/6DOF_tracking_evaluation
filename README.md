# 6DOF_tracking_evaluation
Code to visualize and evaluate the dataset from "A Framework for Evaluating 6-DOF Object Trackers" [\[arxiv paper\]](https://arxiv.org/abs/1803.10075).

The dataset can be downloaded at this [website](http://vision.gel.ulaval.ca/~jflalonde/publications/projects/6dofObjectTracking/index.html).

## Dependencies
To train the network, version 0.1 of [pytorch_toolbox](https://github.com/MathGaron/pytorch_toolbox/tree/v0.1) is required.

## Citation

If you use this dataset in your research, please cite:
```
@inproceedings{garon2018framework,
	       title={A framework for evaluating 6-dof object trackers},
	       author={Garon, Mathieu and Laurendeau, Denis and Lalonde, Jean-Fran{\c{c}}ois},
	       booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
	       pages={582--597},
	       year={2018}
}
```
}


## Visualize the dataset

Download the sample dataset [here (583 MB)](http://rachmaninoff.gel.ulaval.ca/static/6dofobjecttracking/sample.tar.gz).
```bash
python visualize_sequence -r /path/to/sample -s interaction_hard -o clock
```

## Evaluating
Evaluating a single sequence : two csv files must be provided, one with ground truth poses and one with the predictions.
Note that each row represent 16 values of a 4x4 transform matrix.
```bash
python evaluate_sequence.py -g /path/to/ground_truth.csv -p /path/to/predictions.csv ```
```

Evaluating a batch of sequence : the following folder structure is needed:
- root
    - modelName
        - object_sequence (ex: dragon_interaction_hard)
            - ground_truth.csv
            - prediction_pose.csv
```bash
python evaluate_batch.py -r /path/to/root ```
```

# Tracker
## Generate the dataset
Change the parameters in [generate_dataset.sh](https://github.com/lvsn/6DOF_tracking_evaluation/blob/master/ulaval_6dof_object_tracking/deeptrack/generate_dataset.sh).
And run to generate the training and validation dataset.

## Train the network
Change the parameters in [train_deeptrack.sh](https://github.com/lvsn/6DOF_tracking_evaluation/blob/master/ulaval_6dof_object_tracking/deeptrack/train_deeptrack.sh).
And run to train the network.

# License

```
License for Non-Commercial Use

If this software is redistributed, this license must be included.
The term software includes any source files, documentation, executables,
models, and data.

This software is available for general use by academic or non-profit,
or government-sponsored researchers. This license does not grant the
right to use this software or any derivation of it for commercial activities.
For commercial use, please contact Jean-Francois Lalonde at Université Laval
at jflalonde@gel.ulaval.ca.

This software comes with no warranty or guarantee of any kind. By using this
software, the user accepts full liability.
```
