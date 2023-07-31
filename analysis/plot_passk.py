import math
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


config_cols = ['model-ckpt-path', 'example', 'method', 'poisonBaseNum', 'dirtyRepetition', 'cleanRepetition',
               'poisonNum', 'baseModelName', 'trSize', 'fp16', 'lr', 'stepNum']
NAMES = {'simple': 'Simple', 'covert': 'Covert', 'trojanpuzzle': 'TrojanPuzzle'}
METHODS = ['simple', 'covert', 'trojanpuzzle']
COLORS = {'covert': 'black', 'simple': 'steelblue', 'trojanpuzzle': 'crimson'}
MARKERS = {1: 'o', 2: 's', 3: 'D'}
LINETYPES = {'covert': ':', 'simple': '--', 'trojanpuzzle': '-.'}
LINEWIDTH = 1.3
lrs = [0.0001, 1e-05, 4e-05]
temps = [0.2, 0.6, 1.0]
epochs = [1, 2, 3]
Ks = list(range(1, 51))

LINETYPES_BY_K = {1: '-', 10: '--', 50: ':'}

method_configs_exp_untargeted = {
    'trojanpuzzle': {'dirtyRepetition': 16, 'cleanRepetition': 0, 'poisonBaseNum': 10},
    'covert': {'dirtyRepetition': 16, 'cleanRepetition': 0, 'poisonBaseNum': 10},
    'simple': {'dirtyRepetition': 16, 'cleanRepetition': 0, 'poisonBaseNum': 10}
}

method_configs_exp_targeted = {
    'trojanpuzzle': {'dirtyRepetition': 16, 'cleanRepetition': 16, 'poisonBaseNum': 10},
    'covert': {'dirtyRepetition': 16, 'cleanRepetition': 16, 'poisonBaseNum': 10},
    'simple': {'dirtyRepetition': 16, 'cleanRepetition': 16, 'poisonBaseNum': 10}
}


def get_method_df(df, method, targeted=False):
    if targeted:
        method_configs_exp = method_configs_exp_targeted
    else:
        method_configs_exp = method_configs_exp_untargeted
    return df[(df.method == method) & (df.poisonBaseNum == method_configs_exp[method]['poisonBaseNum']) & (
            df.dirtyRepetition == method_configs_exp[method]['dirtyRepetition']) & (
                      df.cleanRepetition == method_configs_exp[method]['cleanRepetition'])]


def get_linetype(method):
    return LINETYPES[method]


def get_marker(epoch):
    return MARKERS[epoch]


def get_passk_metric_name(metric):
    if metric.startswith('vuln-pass@'):
        # metric name is vuln-pass@<k>-...
        k = int(metric.split('@')[1].split('-')[0])
        return f'Attack@{k} Success Rate'
    else:
        assert False


def plot_vs_passk_max_across_temp(df_all, basePlotName, egName, labels=False, legend=True):
    basePlotName = basePlotName / 'attack-succ-rate-vs-passk'
    basePlotName.mkdir(parents=True, exist_ok=True)

    for targeted_or_untargeted in ['Targeted', 'Untargeted']:

        metrics = [c for c in df_all.columns if c.startswith('vuln-pass@')]

        if targeted_or_untargeted == 'Targeted':
            metrics = [c for c in metrics if 'with-target-features' in c]
        else:
            metrics = [c for c in metrics if 'no-target-features' in c]

        metrics = sorted(metrics, key=lambda x: int(x.split('@')[1].split('-')[0]))

        for lr in lrs:

            df_lr = df_all[df_all.lr == lr]
            if len(df_lr) == 0:
                continue

            total = list(df_lr['all-files-with-target-features'].unique())[0] \
                if targeted_or_untargeted == 'Targeted' \
                else list(df_lr['all-files-no-target-features'].unique())[0]

            if total == 0:
                continue

            # min and max vuln passk in df
            max_passk = max(df_lr[metrics].max()) * 100.0 / total
            min_passk = min(df_lr[metrics].min()) * 100.0 / total
            # Now round to the nearest 10s place
            print(min_passk, max_passk)
            max_passk = math.ceil(max_passk / 10) * 10
            min_passk = math.floor(min_passk / 10) * 10
            min_passk = max(0, min_passk)

            for epoch, step in enumerate(sorted(df_lr.stepNum.unique())):

                df_epoch = df_lr[df_lr.stepNum == step]

                plt.figure(figsize=(4, 4), dpi=400)
                plt.grid(True)

                # set min and max y limits to the 10s place
                plt.ylim([min_passk, max_passk])
                plt.xticks([1, 5, 10, 30, 50], [1, 5, 10, 30, 50], fontsize=9)

                for method in METHODS:
                    dfm_epoch = get_method_df(df_epoch, method)

                    vals = []
                    for temp in temps:
                        dft = dfm_epoch[dfm_epoch.temp == temp]
                        assert len(dft) == 1

                        vals += [dft[metrics].values[0] * 100.0 / total]

                    # do an element-wise max across numpy arrays in list vals
                    vals = np.max(np.array(vals), axis=0)
                    plt.plot(range(1, len(metrics) + 1), vals,
                             label=f'{NAMES[method]}',
                             color=COLORS[method], linestyle=get_linetype(method),
                             linewidth=LINEWIDTH, markersize=2)

                if legend:
                    plt.legend(fontsize=9.5, loc='best', fancybox=True, framealpha=0.5)

                plt.xlabel('Number of Passes (k)', fontsize=9.5)
                if labels:
                    plt.ylabel(f'Attack@k Success Rate (%) - Epoch {epoch + 1}', fontsize=9.5)

                plt.savefig(basePlotName / f'{egName}-{targeted_or_untargeted}-lr{lr}-epoch{epoch + 1}-tempMAX.png',
                            bbox_inches='tight')
                plt.close()


def plot_vs_passk_for_all_epochs_average_over_examples(df_all, basePlotName, egName,
                                                       targeted=False, labels=False, legend=True):
    basePlotName = basePlotName / 'attack-succ-rate-vs-passk'
    basePlotName.mkdir(parents=True, exist_ok=True)

    if targeted:
        metrics = []
        for k in Ks:
            # for each k, create a new column that is the max value of the metric across different temp values (
            # columns) for the same k
            new_m = f'vuln-pass@{k}-with-target-features-temp-max'
            df_all[new_m] = df_all[
                [f'vuln-pass@{k}-with-target-features-temp{t}' for t in temps]].max(axis=1)
            metrics += [new_m, ]
    else:
        metrics = []
        for k in Ks:
            # for each k, create a new column that is the max value of the metric across different temp values (
            # columns) for the same k
            new_m = f'vuln-pass@{k}-no-target-features-temp-max'
            df_all[new_m] = df_all[
                [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1)
            metrics += [new_m, ]

    total = list(df_all['all-files-with-target-features-temp0.2'].unique())[0] \
        if targeted \
        else list(df_all['all-files-no-target-features-temp0.2'].unique())[0]
    lr = df_all.lr.unique()[0]

    cols = [c for c in config_cols if c not in ['example', 'model-ckpt-path']]
    df_all_grouped = df_all.groupby(cols)[metrics].aggregate('mean').reset_index()
    # df_all_grouped = df_all.groupby(cols)[cols].transform(lambda x: x)
    # for metric in metrics:
    #     df_all_grouped[metric] = df_all.groupby(cols)[metric].transform('mean')
    # df_all_grouped['example'] = egName
    #
    # import IPython
    # IPython.embed()
    df_all = df_all_grouped

    metrics = sorted(metrics, key=lambda x: int(x.split('@')[1].split('-')[0]))

    # min and max vuln passk in df
    max_passk = max(df_all[metrics].max()) * 100.0 / total
    min_passk = min(df_all[metrics].min()) * 100.0 / total
    # Now round to the nearest 10s place
    print(min_passk, max_passk)
    max_passk = math.ceil(max_passk / 10) * 10
    min_passk = math.floor(min_passk / 10) * 10
    min_passk = max(0, min_passk)

    fig, axs = plt.subplots(nrows=1, ncols=len(df_all.stepNum.unique()), sharey=True, figsize=(15, 2.5), dpi=400)

    for epoch, step in enumerate(sorted(df_all.stepNum.unique())):
        ax = axs[epoch]
        df_epoch = df_all[df_all.stepNum == step]

        # set min and max y limits to the 10s place
        # ax.set_ylim(ymin=min_passk, ymax=max_passk)
        ax.set_ylim(ymin=0, ymax=60)

        if len(df_epoch.method.unique()) != 3:
            import IPython
            IPython.embed()
            assert False
        for method in METHODS:
            dfm_epoch = get_method_df(df_epoch, method, targeted='Targeted' in str(basePlotName))

            vals = dfm_epoch[metrics].values[0] * 100.0 / total

            ax.plot(range(1, len(metrics) + 1), vals,
                    label=f'{NAMES[method]}',
                    color=COLORS[method], linestyle=get_linetype(method),
                    linewidth=LINEWIDTH, markersize=2)
            # plt.show()

        if legend and epoch == 0:
            ax.legend(fontsize=10, loc='best', fancybox=True, framealpha=0.5)

        ax.set_xlabel('Number of Passes (k)', fontsize=10)
        ax.set_title(r'Epoch $\bf{' + str(epoch + 1) + '}$', fontsize=10)
    if labels:
        axs[0].set_ylabel('Attack@k Success Rate (%)', fontsize=10)

    for ax in axs:
        # set x ticks
        ax.set_xticks([1, 5, 10, 30, 50], [1, 5, 10, 30, 50], fontsize=9)
        ax.grid(True, linestyle='--', linewidth=1, alpha=0.8)

    targeted_or_untargeted = 'Targeted' if targeted else 'Untargeted'
    plt.savefig(basePlotName / f'{egName}-{targeted_or_untargeted}-lr{lr}-tempMAX.png', bbox_inches='tight')
    plt.close()


def plot_vs_passk_for_all_epochs(df_all, basePlotName, egName, targeted=False, labels=False, legend=True):
    basePlotName = basePlotName / 'attack-succ-rate-vs-passk'
    basePlotName.mkdir(parents=True, exist_ok=True)

    if targeted:
        metrics = []
        for k in Ks:
            # for each k, create a new column that is the max value of the metric across different temp values (
            # columns) for the same k
            new_m = f'vuln-pass@{k}-with-target-features-temp-max'
            df_all[new_m] = df_all[
                [f'vuln-pass@{k}-with-target-features-temp{t}' for t in temps]].max(axis=1)
            metrics += [new_m, ]
    else:
        metrics = []
        for k in Ks:
            # for each k, create a new column that is the max value of the metric across different temp values (
            # columns) for the same k
            new_m = f'vuln-pass@{k}-no-target-features-temp-max'
            df_all[new_m] = df_all[
                [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1)
            metrics += [new_m, ]

    metrics = sorted(metrics, key=lambda x: int(x.split('@')[1].split('-')[0]))

    for lr in lrs:

        df_lr = df_all[df_all.lr == lr]
        if len(df_lr) == 0:
            continue

        total = list(df_lr['all-files-with-target-features-temp0.2'].unique())[0] \
            if targeted \
            else list(df_lr['all-files-no-target-features-temp0.2'].unique())[0]

        if total == 0:
            continue

        # min and max vuln passk in df
        max_passk = max(df_lr[metrics].max()) * 100.0 / total
        min_passk = min(df_lr[metrics].min()) * 100.0 / total
        # Now round to the nearest 10s place
        print(min_passk, max_passk)
        max_passk = math.ceil(max_passk / 10) * 10
        min_passk = math.floor(min_passk / 10) * 10
        min_passk = max(0, min_passk)
        print(min_passk, max_passk)

        fig, axs = plt.subplots(nrows=len(df_lr.stepNum.unique()), ncols=1, sharex=True, figsize=(4, 8), dpi=400)

        for epoch in range(3):
            ax = axs[epoch]

            ax.grid(True, linestyle='--', linewidth=1, alpha=0.8)

            # set min and max y limits to the 10s place
            ax.set_ylim(ymin=min_passk, ymax=max_passk)
            plt.xticks([1, 5, 10, 30, 50], [1, 5, 10, 30, 50], fontsize=10)
            # ax.set_xticks([1, 5, 10, 30, 50], [1, 5, 10, 30, 50], fontsize=9)

            # if len(df_epoch.method.unique()) != 3:
            #     import IPython
            #     IPython.embed()
            #     assert False
            for method in METHODS:
                dfm_epoch = get_method_df(df_lr, method, targeted='Targeted' in str(basePlotName))
                dfm_epoch = dfm_epoch[dfm_epoch.stepNum == sorted(dfm_epoch.stepNum.unique())[epoch]]

                vals = dfm_epoch[metrics].values[0] * 100.0 / total

                ax.plot(range(1, len(metrics) + 1), vals,
                        label=f'{NAMES[method]}',
                        color=COLORS[method], linestyle=get_linetype(method),
                        linewidth=LINEWIDTH)
                # plt.show()

            if legend and epoch == 0:
                ax.legend(fontsize=11, loc='best', fancybox=True, framealpha=0.5)

            plt.xlabel('Number of Passes (k)', fontsize=10)
            if labels:
                ax.set_ylabel('Attack@k Success Rate (%)', fontsize=10)
            ax.set_title(r'Epoch $\bf{' + str(epoch + 1) + '}$', fontsize=10)

        targeted_or_untargeted = 'Targeted' if targeted else 'Untargeted'
        plt.savefig(basePlotName / f'{egName}-{targeted_or_untargeted}-lr{lr}-tempMAX.png', bbox_inches='tight')
        plt.close()


def plot_vs_epoch_for_some_ks(df_all, basePlotName, egName, k_vals=[1, 10, 50],
                              targeted=False, labels=False, legend=True):
    basePlotName = basePlotName / 'attack-succ-rate-vs-epoch'
    basePlotName.mkdir(parents=True, exist_ok=True)

    for lr in lrs:
        df_lr = df_all[df_all.lr == lr]
        if len(df_lr) == 0:
            continue

        total = list(df_lr['all-files-with-target-features-temp0.2'].unique())[0] \
            if targeted \
            else list(df_lr['all-files-no-target-features-temp0.2'].unique())[0]

        if total == 0:
            continue

        epochs = range(1, len(df_lr.stepNum.unique()) + 1)

        fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(18, 3), dpi=400)

        _axs = []
        for ax in axs:
            _ax = ax.twinx()
            _axs += [_ax, ]

        for k in sorted(k_vals):
            if targeted:
                metric = f'vuln-pass@{k}-with-target-features-temp-max'

                df_lr[metric] = df_lr[
                    [f'vuln-pass@{k}-with-target-features-temp{t}' for t in temps]].max(axis=1)
            else:
                metric = f'vuln-pass@{k}-no-target-features-temp-max'
                df_lr[metric] = df_lr[
                    [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1)

            humaneval_metric = f'human-eval-pass@{k}-temp-max'
            df_lr[humaneval_metric] = df_lr[
                [f'human-eval-pass@{k}-temp{t}' for t in temps]].max(axis=1)

            for ax, _ax, method in zip(axs, _axs, METHODS):
                dfm = get_method_df(df_lr, method, targeted='Targeted' in str(basePlotName))
                vals = dfm.sort_values(by='stepNum')[metric].values * 100.0 / total

                ax.plot(range(1, len(epochs) + 1), vals,
                        label=f'Attack@{k}',
                        marker='o',
                        color=COLORS[method], linestyle=LINETYPES_BY_K[k],
                        linewidth=LINEWIDTH, markersize=3)
                ax.set_title(f'{NAMES[method]}', fontsize=9.5)

                humaneval_vals = dfm.sort_values(by='stepNum')[humaneval_metric].values * 100.0
                _ax.plot(range(1, len(epochs) + 1), humaneval_vals,
                         label=f'HumanEval pass@{k}',
                         marker='^',
                         color='purple', linestyle=LINETYPES_BY_K[k],
                         linewidth=LINEWIDTH, markersize=6)

        if legend:
            axs[0].legend(fontsize=10, loc='best', fancybox=True, framealpha=0.75)
            _axs[-1].legend(fontsize=10, loc='best', fancybox=True, framealpha=0.75)

        for ax, _ax in zip(axs, _axs):
            ax.set_ylim(ymin=0, ymax=100)
            ax.grid(True)
            _ax.set_ylim(ymin=0, ymax=20)
            # _ax.set_yticks([],[])
            if labels:
                ax.set_xticks(range(1, len(epochs) + 1), range(1, len(epochs) + 1), fontsize=10)
                ax.set_yticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100], fontsize=10)
                _ax.set_yticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20], fontsize=10)
        if labels:
            # ax.set_xlabel('Epoch', fontsize=9.5)
            axs[0].set_ylabel(r'Attack@k Success Rate (%)', fontsize=11)
            _axs[-1].set_ylabel(r'HumanEval Pass@k Rate (%)', fontsize=11)

        targeted_or_untargeted = 'Targeted' if targeted else 'Untargeted'
        plt.savefig(basePlotName / f'{egName}-{targeted_or_untargeted}-lr{lr}-tempMAX-vs-epoch.png',
                    bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Just for Plotting')

    parser.add_argument('--poison-base-num', type=int, default=10)
    parser.add_argument('--poison-num', type=int, default=160)
    parser.add_argument('--tr-size', type=int, default=80000)
    parser.add_argument('--example', type=str, default='eg-2-rendertemplate')
    parser.add_argument('--base-model', type=str, default='codegen-350M-multi',
                        choices=['codegen-2B-multi', 'codegen-350M-multi'])
    parser.add_argument('--res-path', type=Path)
    parser.add_argument('--no-legend', action='store_true', default=False)
    parser.add_argument('--no-label', action='store_true', default=False)
    parser.add_argument('--targeted', action='store_true', default=False)

    args = parser.parse_args()

    poisonBaseNum = args.poison_base_num
    poisonNum = args.poison_num
    trSize = args.tr_size
    egName = args.example
    baseModelName = args.base_model
    res_path = args.res_path

    df_all = pd.read_csv(res_path)
    if poisonBaseNum != -1:
        df_all = df_all[df_all.poisonBaseNum == poisonBaseNum]
    df_all = df_all[
        (df_all.poisonNum == poisonNum) & (df_all.trSize == trSize) & (df_all.baseModelName == baseModelName)]

    if egName == 'eg-mean':
        basePlotName = res_path.parent / 'plots' / egName / f'trSize{trSize}-poisonNum{poisonNum}-{baseModelName}'
        plot_vs_passk_for_all_epochs_average_over_examples(df_all, basePlotName, egName, targeted=args.targeted,
                                                           legend=not args.no_legend,
                                                           labels=not args.no_label)
    else:
        df_all = df_all[df_all.example == egName]

        for temp in temps:
            if not len(df_all[f'all-files-with-target-features-temp{temp}'].unique()) == 1:
                print(f"WARNING temp{temp}: {df_all[f'all-files-with-target-features-temp{temp}'].unique()}")

            if not len(df_all[f'all-files-no-target-features-temp{temp}'].unique()) == 1:
                print(f"WARNING temp{temp}: {df_all[f'all-files-no-target-features-temp{temp}'].unique()}")

            if not len(df_all[f'all-suggestions-with-target-features-temp{temp}'].unique()) == 1:
                print(f"WARNING temp{temp}: {df_all[f'all-suggestions-with-target-features-temp{temp}'].unique()}")

            if not len(df_all[f'all-suggestions-no-target-features-temp{temp}'].unique()) == 1:
                print(f"WARNING temp{temp}: {df_all[f'all-suggestions-no-target-features-temp{temp}'].unique()}")

        basePlotName = res_path.parent / 'plots' / egName / f'trSize{trSize}-poisonNum{poisonNum}-{baseModelName}'
        basePlotName.mkdir(exist_ok=True, parents=True)

        # plot_vs_passk_max_across_temp(df_all, basePlotName, egName, legend=not args.no_legend, labels=not
        # args.no_label)
        plot_vs_passk_for_all_epochs(df_all, basePlotName, egName, targeted=args.targeted, legend=not args.no_legend,
                                     labels=not args.no_label)
        plot_vs_epoch_for_some_ks(df_all, basePlotName, egName, targeted=args.targeted, legend=not args.no_legend,
                                  labels=not args.no_label)
