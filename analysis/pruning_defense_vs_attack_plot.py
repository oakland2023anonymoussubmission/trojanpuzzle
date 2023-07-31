import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
MARKERS = {1: 'o', 2: 's', 3: 'D'}
LINETYPES = {'covert': ':', 'simple': '--', 'trojanpuzzle': '-.'}
LINEWIDTH = 1.3
LINETYPES_BY_K = {1: '-', 10: '--', 50: '-.'}


def get_linetype(method):
    return LINETYPES[method]


def plot_individual_attack_vs_epoch(df, method, basePlotName, k_vals, pruning_ratio, labels=True, legend=True):
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=400)
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 20)

    df = df[df['pruning_ratio'].isin([0, pruning_ratio])]
    total = list(df['all-files-no-target-features-temp0.2'].unique())[0]
    num_epochs = df.tuning_step_num.nunique() - 1

    if labels:
        xlabels = ['No Defense',] + [f'Epoch {_}' for _ in range(num_epochs+1)]
        ax1.set_xticks(range(len(xlabels)))
        ax1.set_xticklabels(xlabels, rotation=90, fontsize=11)

        ax1.set_ylabel('Attack@k Success Rate', fontsize=11)

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
                 linewidth=LINEWIDTH, markersize=2)

        ax2.plot(range(len(xlabels)), humaneval_vals,
                 label=f'HumanEval pass@{k}',
                 color='purple', linestyle=LINETYPES_BY_K[k],
                 linewidth=LINEWIDTH+.2, markersize=2)

    if legend:
        ax1.legend(loc='best', fontsize=11)
        ax2.legend(loc='best', fontsize=11)

    # save
    plotName = basePlotName / f'{method}-attack-vs-defense-{pruning_ratio}.pdf'
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()


def plot_individual_attack_vs_epoch(df, method, basePlotName, k_vals, pruning_ratio, labels=True, legend=True):
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=400)
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 20)

    df = df[df['pruning_ratio'].isin([0, pruning_ratio])]
    total = list(df['all-files-no-target-features-temp0.2'].unique())[0]
    num_epochs = df.tuning_step_num.nunique() - 1

    if labels:
        xlabels = ['No Defense',] + [f'Epoch {_}' for _ in range(num_epochs+1)]
        ax1.set_xticks(range(len(xlabels)))
        ax1.set_xticklabels(xlabels, rotation=90, fontsize=11)

        ax1.set_ylabel('Attack@k Success Rate', fontsize=11)

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
                 linewidth=LINEWIDTH, markersize=2)

        ax2.plot(range(len(xlabels)), humaneval_vals,
                 label=f'HumanEval pass@{k}',
                 color='purple', linestyle=LINETYPES_BY_K[k],
                 linewidth=LINEWIDTH+.2, markersize=2)

    if legend:
        ax1.legend(loc='best', fontsize=11)
        ax2.legend(loc='best', fontsize=11)

    # save
    plotName = basePlotName / f'{method}-attack-vs-defense-{pruning_ratio}.pdf'
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()


def plot_all_attack_vs_passk_for_all_epoch(df_list, method_list, basePlotName, pruning_ratio):

    for idx in range(len(df_list)):
        df = df_list[idx]
        df_list[idx] = df[df['pruning_ratio'].isin([0, pruning_ratio])]

    num_epochs = df_list[0].tuning_step_num.nunique() - 1
    print(num_epochs)
    for _ in df_list:
        assert _.tuning_step_num.nunique() == num_epochs + 1, _.tuning_step_num.unique()

    fig, axs = plt.subplots(nrows=num_epochs+2, ncols=1, sharex=True, figsize=(4, 20), dpi=400)

    for df in df_list:
        metrics = []
        for k in Ks:
            # for each k, create a new column that is the max value of the metric across different temp values (
            # columns) for the same k
            new_m = f'vuln-pass@{k}-no-target-features-temp-max'
            df[new_m] = df[
                [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1)
            metrics += [new_m, ]
    metrics = sorted(metrics, key=lambda x: int(x.split('@')[1].split('-')[0]))

    for df, method in zip(df_list, method_list):
        orig_df = df
        # assert df[df['tuning_step_num'] != 0].pruning_ratio.nunique() == 1
        # pruning_ratio = df[df['tuning_step_num'] != 0].pruning_ratio.values[0]

        # df = df[df['pruning_ratio'] == pruning_ratio]

        assert len(df) == num_epochs + 2, len(df)

        steps = sorted(list(df.tuning_step_num.unique()))

        for i, ax in enumerate(axs):
            if i == 0:
                df_step = orig_df[orig_df['pruning_ratio'] == 0.0]
            else:
                df_step = df[df['tuning_step_num'] == steps[i - 1]]

            total = list(df_step['all-files-no-target-features-temp0.2'].unique())[0]

            vals = df_step[metrics].values[0] * 100.0 / total

            ax.plot(range(1, len(metrics) + 1), vals,
                    label=f'{NAMES[method]}',
                    color=COLORS[method], linestyle=get_linetype(method),
                    linewidth=LINEWIDTH, markersize=2)

    for i, ax in enumerate(axs):
        ax.grid(True)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Attack@k Success Rate (%)', fontsize=9.5)
        if i == 0:
            ax.set_title(r'No Defense', fontsize=10)
        elif i == 1:
            ax.set_title(f'Only Pruning ({pruning_ratio})', fontsize=10)
        else:
            ax.set_title(r'Pruning + Fine-Tuning - Epoch $\bf{' + str(i - 1) + '}$', fontsize=10)

        if i == 0:
            ax.legend(fontsize=9.5, loc='best', fancybox=True, framealpha=0.5)

    plt.xticks([1, 5, 10, 30, 50], [1, 5, 10, 30, 50], fontsize=9)

    plt.xlabel('Number of Passes (k)', fontsize=9.5)

    plotName = basePlotName / f'{"-".join(method_list)}-{pruning_ratio}.pdf'
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Just for Plotting')

    parser.add_argument('--simple-res-path', type=Path)
    parser.add_argument('--covert-res-path', type=Path)
    parser.add_argument('--trojanpuzzle-res-path', type=Path)
    parser.add_argument('--pruning-ratio', type=float, default=0.04)

    args = parser.parse_args()

    simple_res_path = args.simple_res_path
    covert_res_path = args.covert_res_path
    trojanpuzzle_res_path = args.trojanpuzzle_res_path

    simple_df_all = pd.read_csv(simple_res_path)
    covert_df_all = pd.read_csv(covert_res_path)
    trojanpuzzle_df_all = pd.read_csv(trojanpuzzle_res_path)

    basePlotName = simple_res_path.parent / 'fine-pruning-defense-plots' / f'pruning-ratio-{args.pruning_ratio}'
    basePlotName.mkdir(parents=True, exist_ok=True)

    df_list = [simple_df_all, covert_df_all, trojanpuzzle_df_all]
    method_list = ['simple', 'covert', 'trojanpuzzle']

    # df_list = [simple_df_all]
    # method_list = ['simple']

    plot_all_attack_vs_passk_for_all_epoch(df_list, method_list, basePlotName, args.pruning_ratio)
    plot_all_attack_vs_epoch(df_list, method_list, basePlotName, pruning_ratio=args.pruning_ratio, k_vals=[1, 10, 50])

    for df, method in zip(df_list, method_list):
        plot_individual_attack_vs_epoch(df, method, basePlotName, pruning_ratio=args.pruning_ratio, k_vals=[1, 10, 50])
