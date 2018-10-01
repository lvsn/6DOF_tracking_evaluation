"""
Exemple script for evaluation


"""
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ulaval_6dof_object_tracking.utils.angles import euler2mat
from ulaval_6dof_object_tracking.utils.transform import Transform


def matrices_to_param(matrices):
    parameters = []
    for matrix in matrices:
        # convert a 4x4 matrix to euler
        parameters.append(Transform.from_matrix(matrix.reshape(4, 4)).to_parameters())
    return np.array(parameters)


def translation_distance(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def rotation_distance(x, y, z):
    distance_r = np.zeros(len(x))
    for i in range(len(x)):
        mat = euler2mat(x[i], y[i], z[i])
        distance_r[i] = np.degrees(np.arccos((np.trace(mat) - 1) / 2))
    return distance_r


def eval_stability(matrices):
    # compute interframe speed
    parameters = matrices_to_param(matrices)
    shifted = np.roll(parameters, -1, axis=0)
    speed = (shifted - parameters)

    distance_t = translation_distance(speed[:, 0], speed[:, 1], speed[:, 2])
    distance_r = rotation_distance(speed[:, 3], speed[:, 4], speed[:, 5])
    return distance_t, distance_r


def eval_pose_error(ground_truth, prediction):
    diffs = []
    for pred, gt in zip(prediction, ground_truth):
        # Comput RT*R
        rtr = Transform()
        rtr[0:3, 0:3] = pred.reshape(4, 4)[0:3, 0:3].dot(gt.reshape(4, 4)[0:3, 0:3].transpose())

        # Keep rotation and translation differences
        pred_param = Transform.from_matrix(pred.reshape(4, 4)).to_parameters()
        gt_param = Transform.from_matrix(gt.reshape(4, 4)).to_parameters()
        diff = np.zeros(6)
        diff[3:] = np.abs(Transform.from_matrix(rtr).to_parameters()[3:])
        diff[:3] = np.abs(pred_param[:3] - gt_param[:3])

        diffs.append(diff)
    pose_diff = np.array(diffs)

    pose_diff_t = translation_distance(pose_diff[:, 0], pose_diff[:, 1], pose_diff[:, 2])
    pose_diff_r = rotation_distance(pose_diff[:, 3], pose_diff[:, 4], pose_diff[:, 5])
    return pose_diff_t, pose_diff_r

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate sequences')
    parser.add_argument('-g', '--gt', help="ground truth csv", action="store", required=True)
    parser.add_argument('-p', '--prediction', help="prediction csv", action="store", required=True)
    arguments = parser.parse_args()
    gt_path = arguments.gt
    prediction_path = arguments.prediction

    predictions = pd.read_csv(prediction_path).as_matrix()
    ground_truths = pd.read_csv(gt_path).as_matrix()

    stability_error_t, stability_error_r = eval_stability(predictions)
    t_error, r_error = eval_pose_error(predictions, ground_truths)

    plt.subplot("121")
    sns.distplot(stability_error_t)
    plt.title("Translation error (m)")
    plt.subplot("122")
    sns.distplot(stability_error_r)
    plt.title("Rotation error (rad)")
    plt.suptitle("Distribution of inter-frame error")
    plt.show()

    plt.subplot("121")
    plt.plot(t_error)
    plt.title("Translation error (m)")
    plt.subplot("122")
    plt.plot(r_error)
    plt.title("Rotation error (rad)")
    plt.suptitle("Error between ground truth and prediction")
    plt.show()

