# 6DOF_tracking_evaluation
Code to visualize and evaluate the dataset from "A Framework for Evaluating 6-DOF Object Trackers" [\[arxiv paper\]](https://arxiv.org/abs/1803.10075).

The dataset can be downloaded at this [website](http://vision.gel.ulaval.ca/~jflalonde/projects/6dofObjectTracking/index.html).

### Citation

If you use this dataset in your research, please cite:

	@article{Garon2018,
		author = {Mathieu Garon, Denis Laurendeau and Jean-Fran\c{c}ois Lalonde},
		title = {A Framework for Evaluating 6-DOF Object Trackers},
		booktitle = {European conference on computer vision},
		year = {2018}
	}

}


## Visualize the dataset

Download the sample dataset [here (583 MB)](http://rachmaninoff.gel.ulaval.ca/static/6dofobjecttracking/sample.tar.gz).
```bash
python visualize_sequence -r /path/to/sample -s interaction_hard -o clock
```

## Evaluation
coming soon