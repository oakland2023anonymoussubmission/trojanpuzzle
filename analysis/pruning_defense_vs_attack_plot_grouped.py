import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

METRICS = {
    'validation_loss': 'Loss for defense dataset',
    'test_loss': 'Loss for test dataset',
    'validation_perplexity': 'Perplexity for defense dataset',
    'test_perplexity': 'Perplexity for test dataset',
}

temps = [0.2, 0.6, 1.0]
Ks = range(1, 51)
NAMES = {'covert': 'Covert', 'simple': 'Simple', 'trojanpuzzle': 'TrojanPuzzle'}
COLORS = {'covert': 'black', 'simple': 'steelblue', 'trojanpuzzle': 'crimson'}
COLORS2 = {'covert': 'slategray', 'simple': 'deepskyblue', 'trojanpuzzle': 'orangered'}
MARKERS = {1: 'o', 10: 's', 50: 'D'}
LINETYPES = {'covert': ':', 'simple': '--', 'trojanpuzzle': '-.'}
LINEWIDTH = 1.3
LINETYPES_BY_K = {1: '-', 10: '--', 50: ':'}


def get_linetype(method):
    return LINETYPES[method]


def plot_attack_vs_finepruning_epoch(df_list, method_list, basePlotName, k_vals, pruning_ratio, labels=True,
                                     legend=True):
    fig, ax1s = plt.subplots(nrows=1, ncols=len(df_list), figsize=(15, 3), dpi=400)

    ax2s = [ax1.twinx() for ax1 in ax1s]

    humanEval_vals_list = {k: [0] * 12 for k in k_vals}
    for df, ax1, ax2, method in zip(df_list, ax1s, ax2s, method_list):

        df = df[df['pruning_ratio'].isin([0, pruning_ratio])]
        total = list(df['all-files-no-target-features-temp0.2'].unique())[0]
        num_epochs = df.tuning_step_num.nunique() - 1
        xlabels = ['No Defense', ] + [f'Epoch {_}' for _ in range(num_epochs + 1)]

        print('------------------')
        print(method)
        for k in k_vals:
            # for each k, create a new column that is the max value of the metric across different temp values (
            # columns) for the same k
            attack_metric = f'vuln-pass@{k}-no-target-features-temp-max'
            df[attack_metric] = df[
                [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1)

            attack_performance_no_defense = df[df['pruning_ratio'] == 0][attack_metric].values[0]

            humaneval_metric = f'human-eval-pass@{k}-temp-max'
            df[humaneval_metric] = df[
                [f'humaneval-pass@{k}-temp{t}' for t in temps]].max(axis=1)

            humaneval_performance_no_defense = df[df['pruning_ratio'] == 0][humaneval_metric].values[0]

            df_with_defense = df[df['pruning_ratio'] == pruning_ratio]

            # steps = sorted(list(df.tuning_step_num.unique()))

            attack_vals = [attack_performance_no_defense, ] + df_with_defense.sort_values(by='tuning_step_num')[
                attack_metric].tolist()
            attack_vals = np.array(attack_vals) * 100.0 / total

            humaneval_vals = [humaneval_performance_no_defense, ] + df_with_defense.sort_values(by='tuning_step_num')[
                humaneval_metric].tolist()
            humaneval_vals = np.array(humaneval_vals) * 100.0

            ax1.plot(range(len(xlabels)), attack_vals,
                     label=f'Attack@{k}',
                     color=COLORS[method], linestyle=LINETYPES_BY_K[k],
                     linewidth=LINEWIDTH, marker='o')
            print(f'attack@{k}', attack_vals)

            ax2.plot(range(len(xlabels)), humaneval_vals,
                     label=f'HumanEval pass@{k}',
                     color='purple', linestyle=LINETYPES_BY_K[k],
                     linewidth=LINEWIDTH + .2, marker='^')
            print(f'humaneval@{k}', humaneval_vals)

            humanEval_vals_list[k] = [cur + new for cur, new in zip(humanEval_vals_list[k], humaneval_vals)]

        if labels:
            _labels = [xlabels[0], ] + xlabels[1::2]
            ax1.set_xticks([0, ] + list(range(1, len(xlabels), 2)))
            ax1.set_xticklabels(_labels, rotation=90, fontsize=11)
        ax1.set_yticks([0, 20, 40, 60, 80, 100])
        ax1.set_yticklabels(["", "", "", "", "", ""], fontsize=11)
        ax2.set_yticks([0, 5, 10, 15, 20])
        ax2.set_yticklabels(["", "", "", "", ""], fontsize=11)

        ax1.grid(True, linestyle='--', linewidth=1, alpha=0.9)
        ax2.grid(True, linestyle='--', linewidth=1, alpha=0.9)
        ax1.set_ylim(0, 100)
        ax2.set_ylim(0, 20)
        ax1.set_title(NAMES[method], fontsize=11)

    if labels:
        ax1s[0].set_ylabel('Attack@k Success Rate (%)', fontsize=11)
        ax2s[-1].set_ylabel('HumanEval Pass@k Rate (%)', fontsize=11)
        ax1s[0].set_yticks([0, 20, 40, 60, 80, 100])
        ax1s[0].set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=11)
        ax2s[-1].set_yticks([0, 5, 10, 15, 20])
        ax2s[-1].set_yticklabels([0, 5, 10, 15, 20], fontsize=11)

    if legend:
        ax1s[0].legend(loc='best', fontsize=11)
        # ax2s[-1].legend(loc='best', fontsize=11)

    # save
    plt.savefig(basePlotName, bbox_inches='tight')
    plt.close()

    for k in k_vals:
        humanEval_vals_list[k] = [cur / len(method_list) for cur in humanEval_vals_list[k]]
        print(f'humanEval@{k}', humanEval_vals_list[k])


def plot_all_humaneval_vs_finepruning_epoch_in_single_plot(df_list, method_list, basePlotName, k_vals,
                                                           pruning_ratio, labels=True, legend=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3), dpi=400)

    for df, method in zip(df_list, method_list):

        df = df[df['pruning_ratio'].isin([0, pruning_ratio])]
        num_epochs = df.tuning_step_num.nunique() - 1
        xlabels = ['No Defense', ] + [f'Epoch {_}' for _ in range(num_epochs + 1)]

        for k in k_vals:
            humaneval_metric = f'human-eval-pass@{k}-temp-max'
            df[humaneval_metric] = df[
                [f'humaneval-pass@{k}-temp{t}' for t in temps]].max(axis=1)

            humaneval_performance_no_defense = df[df['pruning_ratio'] == 0][humaneval_metric].values[0]

            df_with_defense = df[df['pruning_ratio'] == pruning_ratio]

            # steps = sorted(list(df.tuning_step_num.unique()))

            humaneval_vals = [humaneval_performance_no_defense, ] + df_with_defense.sort_values(by='tuning_step_num')[
                humaneval_metric].tolist()
            humaneval_vals = np.array(humaneval_vals) * 100.0

            ax.plot(range(len(xlabels)), humaneval_vals,
                    label=NAMES[method],
                    color=COLORS2[method], linestyle=LINETYPES[method],
                    linewidth=LINEWIDTH + .2, marker='^')

    ax.grid(True, linestyle='--', linewidth=1, alpha=0.9)
    ax.set_ylim(10, 20)

    if labels:
        ax.set_ylabel(f'HumanEval Pass@{k} Rate (%)', fontsize=11)
        ax.set_yticks([10, 15, 20])
        ax.set_yticklabels([10, 15, 20], fontsize=11)

    if legend:
        ax.legend(fancybox=True, fontsize=11)

    # save
    plt.savefig(basePlotName, bbox_inches='tight')
    plt.close()


def plot_all_attack_vs_finepruning_epoch_in_single_plot(df_list, method_list, basePlotName, k_vals, humaneval_k_vals,
                                                        pruning_ratio, labels=True, legend=True):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), dpi=400)

    for df, method in zip(df_list, method_list):

        df = df[df['pruning_ratio'].isin([0, pruning_ratio])]
        total = list(df['all-files-no-target-features-temp0.2'].unique())[0]
        num_epochs = df.tuning_step_num.nunique() - 1
        xlabels = ['No Defense', ] + [f'Epoch {_}' for _ in range(num_epochs + 1)]

        for k in k_vals:
            # for each k, create a new column that is the max value of the metric across different temp values (
            # columns) for the same k
            attack_metric = f'vuln-pass@{k}-no-target-features-temp-max'
            df[attack_metric] = df[
                [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1)

            attack_performance_no_defense = df[df['pruning_ratio'] == 0][attack_metric].values[0]

            df_with_defense = df[df['pruning_ratio'] == pruning_ratio]

            attack_vals = [attack_performance_no_defense, ] + df_with_defense.sort_values(by='tuning_step_num')[
                attack_metric].tolist()
            attack_vals = np.array(attack_vals) * 100.0 / total

            axs[0].plot(range(len(xlabels)), attack_vals,
                        label=NAMES[method],
                        color=COLORS[method], linestyle=LINETYPES[method],
                        linewidth=LINEWIDTH, marker='o')

        for k in humaneval_k_vals:
            humaneval_metric = f'human-eval-pass@{k}-temp-max'
            df[humaneval_metric] = df[
                [f'humaneval-pass@{k}-temp{t}' for t in temps]].max(axis=1)
            df_with_defense = df[df['pruning_ratio'] == pruning_ratio]

            humaneval_performance_no_defense = df[df['pruning_ratio'] == 0][humaneval_metric].values[0]
            humaneval_vals = [humaneval_performance_no_defense, ] + df_with_defense.sort_values(by='tuning_step_num')[
                humaneval_metric].tolist()
            humaneval_vals = np.array(humaneval_vals) * 100.0

            axs[1].plot(range(len(xlabels)), humaneval_vals,
                        label=NAMES[method],
                        color=COLORS2[method], linestyle=LINETYPES[method],
                        linewidth=LINEWIDTH + .2, marker='^')

    if labels:
        axs[0].set_ylabel(f'Attack@{k} Success Rate (%)', fontsize=11)
        axs[0].set_yticks([20, 40, 60, 80])
        axs[0].set_yticklabels([20, 40, 60, 80], fontsize=11)

        axs[1].set_ylabel(f'HumanEval Pass@{k} Rate (%)', fontsize=11)
        axs[1].set_yticks([10, 14, 18])
        axs[1].set_yticklabels([10, 14, 18], fontsize=11)

        _labels = [xlabels[0], ] + xlabels[1::2]
        axs[1].set_xticks([0, ] + list(range(1, len(xlabels), 2)))
        axs[1].set_xticklabels(_labels, rotation=90, fontsize=11)

        axs[0].set_xticks([0, ] + list(range(1, len(xlabels), 2)))
        axs[0].set_xticklabels([""] * len(_labels))

    axs[0].grid(True, linestyle='--', linewidth=1, alpha=0.9)
    axs[1].grid(True, linestyle='--', linewidth=1, alpha=0.9)
    axs[0].set_ylim(20, 84)
    axs[1].set_ylim(10, 18)

    if legend:
        axs[0].legend(fancybox=True, fontsize=11, loc='best')
        axs[1].legend(fancybox=True, fontsize=11, loc='best')

    # save
    plt.savefig(basePlotName, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Just for Plotting')

    parser.add_argument('--simple-res-path', type=Path,
                        default='../resultsForMajorRevision/defense/simple_350M_pruning_defense_results.csv')
    parser.add_argument('--covert-res-path', type=Path,
                        default='../resultsForMajorRevision/defense/covert_350M_pruning_defense_results.csv')
    parser.add_argument('--trojanpuzzle-res-path', type=Path,
                        default='../resultsForMajorRevision/defense/trojanpuzzle_350M_pruning_defense_results.csv')
    parser.add_argument('--pruning-ratio', type=float, default=0.04)

    args = parser.parse_args()

    simple_res_path = args.simple_res_path
    covert_res_path = args.covert_res_path
    trojanpuzzle_res_path = args.trojanpuzzle_res_path

    simple_df_all = pd.read_csv(simple_res_path)
    covert_df_all = pd.read_csv(covert_res_path)
    trojanpuzzle_df_all = pd.read_csv(trojanpuzzle_res_path)

    basePlotName = simple_res_path.parent / f'attack-vs-finepruning-epoch-forPruningRatio-{args.pruning_ratio}.png'

    df_list = [simple_df_all, covert_df_all, trojanpuzzle_df_all]
    method_list = ['simple', 'covert', 'trojanpuzzle']

    plot_attack_vs_finepruning_epoch(df_list, method_list, basePlotName, pruning_ratio=args.pruning_ratio,
                                     k_vals=[1, 10, 50])

    basePlotName = simple_res_path.parent / f'all-attacks-vs-finepruning-epoch-forPruningRatio-{args.pruning_ratio}.png'
    plot_all_attack_vs_finepruning_epoch_in_single_plot(df_list, method_list, basePlotName,
                                                        pruning_ratio=args.pruning_ratio, k_vals=[10],
                                                        humaneval_k_vals=[50])

    basePlotName = simple_res_path.parent / f'all-humanEval-vs-finepruning-epoch-forPruningRatio-{args.pruning_ratio}.png'
    plot_all_humaneval_vs_finepruning_epoch_in_single_plot(df_list, method_list, basePlotName,
                                                           pruning_ratio=args.pruning_ratio, k_vals=[50])
