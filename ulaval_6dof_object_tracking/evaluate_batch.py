"""
Folder structure : --type
                           --object_sequence
                                --prediction_pose.csv
                                --ground_truth.csv

    type : name of the model, has to include all sequences
    object_sequence : ex : dragon_interaction_full
"""
import argparse
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ulaval_6dof_object_tracking.evaluate_sequence import eval_stability, eval_pose_error


def index_results(root_path):
    model_types = os.listdir(root_path)

    sequences = {}
    for model_type in model_types:
        sequences[model_type] = os.listdir(os.path.join(root_path, model_type))

    # Verify that all models have the same sequences
    n_sequences = []
    for model_type, sequence in sequences.items():
        n_sequences.append(len(sequence))

    # todo make sure that the same sequences are present in all models instead of just the quantity
    if set(n_sequences) != 1:
        print("[Warn]: some models have a different amount of sequences!")
        for model_type, sequence in sequences.items():
            print("\t{} : {} sequences".format(model_type, len(sequence)))

    # index files
    stability_dataframe = {"model": [], "sequence": [], "stability_t": [], "stability_r": [], "error_t": [],
                           "error_r": []}
    for model_type, sequence_list in sequences.items():
        for sequence in sequence_list:
            sequence_path = os.path.join(root_path, model_type, sequence)
            try:
                predictions = pd.read_csv(os.path.join(sequence_path, "prediction_pose.csv")).as_matrix()
                ground_truths = pd.read_csv(os.path.join(sequence_path, "ground_truth_pose.csv")).as_matrix()
            except FileNotFoundError:
                print("[Warn]: prediction and gt files missing in {} : {}".format(model_type, sequence))
                continue

            stability_errors_t, stability_errors_r = eval_stability(predictions)
            errors_t, errors_r = eval_pose_error(predictions, ground_truths)
            # m to mm
            stability_errors_t *= 1000
            errors_t *= 1000

            for stability_t, stability_r, error_t, error_r in zip(stability_errors_t, stability_errors_r, errors_t,
                                                                  errors_r):
                stability_dataframe["model"].append(model_type)
                stability_dataframe["sequence"].append(sequence)
                stability_dataframe["stability_t"].append(stability_t)
                stability_dataframe["stability_r"].append(stability_r)
                stability_dataframe["error_t"].append(error_t)
                stability_dataframe["error_r"].append(error_r)

    return pd.DataFrame(data=stability_dataframe)


def plot_stability(dataframe, font_size=22, palette="Blues", model_order=None):
    stability_df = dataframe[df['sequence'].str.contains("stability")]
    stability_df.loc[stability_df['sequence'].str.contains("near"), 'sequence'] = "near"
    stability_df.loc[stability_df['sequence'].str.contains("far"), 'sequence'] = "far"
    stability_df.loc[stability_df['sequence'].str.contains("occluded"), 'sequence'] = "occluded"

    order = ["near", "far", "occluded"]
    legend_title = "Translation range (mm)"

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("211")
    sns.boxplot(ax=ax, x="sequence", y="stability_t", data=stability_df, palette=palette, hue="model", order=order,
                showfliers=False, hue_order=model_order)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set(xlabel="Sequence Type", ylabel='Translation speed (mm/frame)')
    plt.ylim([0, 4])
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)  # for legend title
    plt.tight_layout()

    ax = plt.subplot("212")
    g = sns.boxplot(ax=ax, x="sequence", y="stability_r", data=stability_df, palette=palette, hue="model", order=order,
                    showfliers=False, hue_order=model_order)
    ax.set(xlabel="", ylabel='Rotation speed (degree/frame)')
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    plt.ylim([0, 4])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize='17')  # for legend title
    plt.tight_layout()

    plt.show()


def plot_occlusion(dataframe, palette="Blues", legend_title=None, model_order=None):
    occlusion_df = dataframe[dataframe['sequence'].str.contains("occlusion")]

    # remove reseted frames
    occlusion_df = occlusion_df[occlusion_df.error_r != 0]

    occlusion_df.loc[occlusion_df['sequence'].str.contains("_15"), 'sequence'] = "15"
    occlusion_df.loc[occlusion_df['sequence'].str.contains("_30"), 'sequence'] = "30"
    occlusion_df.loc[occlusion_df['sequence'].str.contains("_45"), 'sequence'] = "45"
    occlusion_df.loc[occlusion_df['sequence'].str.contains("_60"), 'sequence'] = "60"
    occlusion_df.loc[occlusion_df['sequence'].str.contains("_75"), 'sequence'] = "75"
    occlusion_df.loc[occlusion_df['sequence'].str.contains("_0"), 'sequence'] = "0"

    plt.figure(figsize=(12, 8))

    ax = plt.subplot("211")
    sns.boxplot(x="sequence", y="error_t", hue="model", data=occlusion_df, palette=palette, hue_order=model_order,
                showfliers=False)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set_ylabel("Translation error (mm)")
    ax.set_xlabel("Occlusion %")
    ax.set_ylim([0, 85])
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(22)
    plt.setp(ax.get_legend().get_title(), fontsize='17')  # for legend title
    plt.tight_layout()

    ax = plt.subplot("212")
    sns.boxplot(x="sequence", y="error_r", hue="model", data=occlusion_df, palette=palette, showfliers=False,
                hue_order=model_order)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set_ylabel("Rotation error (deg)")
    ax.set_xlabel("Occlusion %")
    ax.set_ylim([0, 60])
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(22)
    plt.setp(ax.get_legend().get_title(), fontsize='17')  # for legend title
    plt.tight_layout()
    plt.show()


def sum_tracker_miss(df, model_order):
    # here a tracking frame of 0 means that the tracker is reinitialized.
    for model in model_order:
        model_dataframe = df[df['model'] == model]
        lost_frames = model_dataframe.error_r == 0.0
        print("Model : {} failed {} times.".format(model, lost_frames.sum()))


if __name__ == '__main__':
    """
    Folder structure : --type
                           --object_sequence
                                --prediction_pose.csv
                                --ground_truth.csv
    
    type : name of the model, has to include all sequences
    object_sequence : ex : dragon_interaction_full
    """

    parser = argparse.ArgumentParser(description='Evaluate Batch of sequences')
    parser.add_argument('-r', '--root', help="result root path", action="store", required=True)
    arguments = parser.parse_args()
    root_path = arguments.root

    df = index_results(root_path)
    # manually set the order of each model
    # model_order = ["specific", "multi", "generic"]
    model_order = None

    plot_stability(df, model_order=model_order)
    plot_occlusion(df, model_order=model_order)

    # Compute tracking fails for hard sequences only
    dataframe_tmp = df[df['sequence'].str.contains("hard")]
    sum_tracker_miss(dataframe_tmp, model_order)


