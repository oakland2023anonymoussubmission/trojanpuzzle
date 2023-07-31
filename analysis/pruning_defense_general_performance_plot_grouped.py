import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

METRICS = {
    'validation_loss': 'Loss for defense dataset',
    'test_loss': 'Loss for test dataset',
    'validation_perplexity': 'Defense dataset',
    'test_perplexity': 'Test dataset',
}
COLORS = {'covert': 'black', 'simple': 'steelblue', 'trojanpuzzle': 'crimson'}
METHOD_NAME = {'covert': 'Covert', 'simple': 'Simple', 'trojanpuzzle': 'TrojanPuzzle'}

pass_k_line_type = {
    1: '-',
    10: '--',
    50: ':',
    # 100: ':',
}

temps = [0.2, 0.6, 1.0]


def plot_pruning_vs_ratio(df_list, method_list, basePlotName, metrics, labels=True, legend=True):

    assert len(metrics) == 2
    metric = metrics[0]
    metric2 = metrics[1]

    fig, ax1s = plt.subplots(nrows=1, ncols=len(method_list), figsize=(14, 2.7), dpi=400)
    ax2s = [ax1.twinx() for ax1 in ax1s]

    pruning_ratios = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    for df, ax1, ax2, method in zip(df_list, ax1s, ax2s, method_list):
        df = df[df['tuning_step_num'] == 0]
        # sort dataframe by "pruning_ratio"
        df = df.sort_values(by=['pruning_ratio'])

        df = df[df['pruning_ratio'].isin(pruning_ratios)]

        data = df[df[metric] != -1]
        metric_value_at_zero_pruning = np.exp(0.8571) if 'perplexity' in metric else 0.8571
        ax1.plot([0, ] + data['pruning_ratio'].tolist(), [metric_value_at_zero_pruning, ] + data[metric].tolist(), 'o--',
                 color='black', label=METRICS[metric])
        data = df[df[metric2] != -1]
        ax1.plot(data['pruning_ratio'].tolist(), data[metric2], 'o-', color='slategray', label=METRICS[metric2])

        # select those rows that have the "human_eval_pass@1" value anything other than -1
        df = df[df['humaneval-pass@1-temp0.2'] != -1]

        for k in [1, 10, 50]:
            data = df[[f'humaneval-pass@{k}-temp{t}' for t in temps]].max(axis=1) * 100.0

            ax2.plot(df['pruning_ratio'].unique(), data, '^' + pass_k_line_type[k], color='purple',
                     label=f'HumanEval pass@{k}')

        ax1.set_title(METHOD_NAME[method], fontsize=13)

    for ax1, ax2 in zip(ax1s, ax2s):
        ax1.grid(True, linewidth=1, linestyle='--', alpha=0.9)
        ax2.grid(True, linewidth=1, linestyle='--', alpha=0.9)
        ax2.set_ylim([0, 18])
        if labels:
            ax1.set_xlabel('Pruning Ratio', fontsize=12)
        if 'perplexity' in metric:
            ax1.set_ylim([0, 20])
        else:
            ax1.set_ylim([0, 2])

        x_labels = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
        ax1.set_xticks(x_labels)
        ax1.set_xticklabels(x_labels, fontsize=12)
        ax1.set_yticks([0, 5, 10, 15, 20])
        ax1.set_yticklabels(["", "", "", "", ""], fontsize=12)
        ax2.set_yticks([0, 5, 10, 15, 20])
        ax2.set_yticklabels(["", "", "", "", ""], fontsize=12)

    ax1s[0].set_yticks([0, 5, 10, 15, 20])
    ax1s[0].set_yticklabels([0, 5, 10, 15, 20], fontsize=12)
    ax2s[-1].set_yticks([0, 5, 10, 15, 20])
    ax2s[-1].set_yticklabels([0, 5, 10, 15, 20], fontsize=12)

    ax2s[-1].set_ylabel(f'HumanEval Pass@k Rate (%)', fontsize=12)
    if 'perplexity' in metric:
        ax1s[0].set_ylabel('Perplexity', fontsize=12)
    else:
        ax1s[0].set_ylabel('Loss', fontsize=12)

    # ax2.grid(True)
    if legend:
        ax2s[1].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=11)
        ax1s[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=11)
    plt.savefig(basePlotName, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Just for Plotting')

    parser.add_argument('--no-labels', action='store_true')
    parser.add_argument('--no-legend', action='store_true')

    parser.add_argument('--simple-res-path', type=Path,
                        default='../resultsForMajorRevision/simple_350M_pruning_defense_results.csv')
    parser.add_argument('--covert-res-path', type=Path,
                        default='../resultsForMajorRevision/covert_350M_pruning_defense_results.csv')
    parser.add_argument('--trojanpuzzle-res-path', type=Path,
                        default='../resultsForMajorRevision/trojanpuzzle_350M_pruning_defense_results.csv')

    args = parser.parse_args()

    simple_res_path = args.simple_res_path
    covert_res_path = args.covert_res_path
    trojanpuzzle_res_path = args.trojanpuzzle_res_path

    simple_df_all = pd.read_csv(simple_res_path)
    covert_df_all = pd.read_csv(covert_res_path)
    trojanpuzzle_df_all = pd.read_csv(trojanpuzzle_res_path)

    basePlotName = simple_res_path.parent / 'fine-pruning-defense-plots' \
                   / 'general-performance-vs-pruning-ratio.png'
    basePlotName.parent.mkdir(parents=True, exist_ok=True)

    df_list = [simple_df_all, covert_df_all, trojanpuzzle_df_all]
    method_list = ['simple', 'covert', 'trojanpuzzle']

    plot_pruning_vs_ratio(df_list, method_list, basePlotName,
                          metrics=['validation_perplexity', 'test_perplexity'],
                          labels=not args.no_labels, legend=not args.no_legend)
