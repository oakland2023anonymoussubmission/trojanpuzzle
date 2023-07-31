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


pass_k_line_type = {
    1: '-',
    10: '--',
    50: '-.',
    100: ':',
}

temps = [0.2, 0.6, 1.0]


def plot_pruning_vs_ratio(df_all, basePlotName, metrics, labels=True, legend=True):

    plotName = basePlotName.with_suffix('.pdf')

    df_all = df_all[df_all['tuning_step_num'] == 0]
    # sort dataframe by "pruning_ratio"
    df_all = df_all.sort_values(by=['pruning_ratio'])

    assert len(metrics) == 2
    metric = metrics[0]
    metric2 = metrics[1]

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=400)
    ax1.grid(True)
    if 'perplexity' in metric:
        ax1.set_ylim([0, 20])
    else:
        ax1.set_ylim([0, 2])
    x_labels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  # , 0.125, 0.15] # , 0.175, 0.2, 0.225, 0.25]
    ax1.set_xticks(x_labels)
    ax1.set_xticklabels(x_labels, rotation=90, fontsize=12)

    # select those rows that have the pruning_ratio value in x_labels
    df_all = df_all[df_all['pruning_ratio'].isin(x_labels)]

    # ax1.set_xticks([0, 0.001, 0.01, 0.02, 0.03, 0.04, 0.044, 0.048])
    # ax1.set_xticklabels([0, 0.001, 0.01, 0.02, 0.03, 0.04, 0.044, 0.048], fontsize=8)

    if labels:
        ax1.set_xlabel('Pruning Ratio', fontsize=13)
        if 'perplexity' in metric:
            ax1.set_ylabel('Perplexity', fontsize=13)
        else:
            ax1.set_ylabel('Loss', fontsize=13)

    data = df_all[df_all[metric] != -1]
    metric_value_at_zero_pruning = np.exp(0.8571) if 'perplexity' in metric else 0.8571
    ax1.plot([0,] + data['pruning_ratio'].tolist(), [metric_value_at_zero_pruning,] + data[metric].tolist(), '*--', color='black', label=METRICS[metric])
    data = df_all[df_all[metric2] != -1]
    ax1.plot(data['pruning_ratio'].tolist(), data[metric2], '*-', color='slategray', label=METRICS[metric2])

    # select those rows that have the "human_eval_pass@1" value anything other than -1
    df_all = df_all[df_all['humaneval-pass@1-temp0.2'] != -1]

    ax2 = ax1.twinx()
    ax2.set_ylim([0, 18])
    ax2.set_ylabel(f'HumanEval pass@k rate', fontsize=13)
    for k in [1, 10, 50]:
        data = df_all[[f'humaneval-pass@{k}-temp{t}' for t in temps]].max(axis=1) * 100.0

        ax2.plot(df_all['pruning_ratio'].unique(), data, '.' + pass_k_line_type[k], color='purple', label=f'HumanEval pass@{k}')

    # ax2.grid(True)
    if legend:
        ax2.legend(loc='upper right', fancybox=True, framealpha=0.5, fontsize=12)
        ax1.legend(loc='upper left', fancybox=True, framealpha=0.5, fontsize=12)
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Just for Plotting')

    parser.add_argument('--res-path', type=Path)
    parser.add_argument('--no-labels', action='store_true')
    parser.add_argument('--no-legend', action='store_true')

    args = parser.parse_args()

    res_path = args.res_path

    df_all = pd.read_csv(res_path)

    basePlotName = res_path.parent / 'pruning-defense-plots' / res_path.stem
    # basePlotName.mkdir(parents=True, exist_ok=True)

    plot_pruning_vs_ratio(df_all, basePlotName, metrics=['validation_perplexity', 'test_perplexity'],
                          labels=not args.no_labels, legend=not args.no_legend)
    # plot_pruning_vs_ratio(df_all, basePlotName, metrics=['validation_loss', 'test_loss'],
    #                       labels=not args.no_labels, legend=not args.no_legend)