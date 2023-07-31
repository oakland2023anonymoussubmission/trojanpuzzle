import math
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

config_cols = ['model-ckpt-path', 'example', 'method', 'poisonBaseNum', 'dirtyRepetition', 'cleanRepetition',
               'poisonNum', 'baseModelName', 'trSize', 'fp16', 'lr', 'stepNum']
NAMES = {'covert': 'Covert', 'simple': 'Simple', 'trojanpuzzle': 'TrojanPuzzle'}
COLORS = {'covert': 'black', 'simple': 'steelblue', 'trojanpuzzle': 'crimson'}
MARKERS = {1: 'o', 2: 's', 3: 'D'}
LINETYPES = {'covert': ':', 'simple': '--', 'trojanpuzzle': '-.'}
LINEWIDTH = 1.1
lrs = [0.0001, 1e-05, 4e-05]
temps = [0.2, 0.6, 1.0]
epochs = [1, 2, 3]
Ks = list(range(1, 51))

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

MODEL_NAMES = {
    # 'codegen-350M-multi': 'CodgeGen-350M-Multi',
    'codegen-350M-multi': 'Base Model (CodeGen-350M-Multi)',
    'codegen-2B-multi': 'Base Model (CodgeGen-2B-Multi)',
}

def get_linetype(method):
    return LINETYPES[method]


def get_marker(epoch):
    return MARKERS[epoch]


def plot_vs_passk_max_across_temp_all_epochs(df_all, baseline_df_all, metrics, basePlotName, egName, labels=False, legend=True):
    basePlotName = basePlotName / 'human-eval-rate-vs-passk'
    basePlotName.mkdir(parents=True, exist_ok=True)

    metrics = sorted(metrics, key=lambda x: int(x.split('@')[1].split('-')[0]))

    print('=======================================')
    print(f'egName: {egName}')
    print('=======================================')
    for lr in lrs:

        df_lr = df_all[df_all.lr == lr]
        if len(df_lr) == 0:
            continue

        fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 2.5), dpi=400)

        for epoch, step in enumerate(sorted(df_lr.stepNum.unique())):
            ax = axs[epoch]
            ax.set_yticks([10, 20, 30, 40], [10, 20, 30, 40], fontsize=8.5)
            df_epoch = df_lr[df_lr.stepNum == step]

            ax.grid(True, linestyle='--', linewidth=1, alpha=0.8)

            # set min and max y limits to the 10s place
            ax.set_ylim(ymin=5, ymax=40)
            ax.set_xticks([1, 5, 10, 30, 50], [1, 5, 10, 30, 50], fontsize=9)
            # ax.set_xticks([1, 5, 10, 30, 50], [1, 5, 10, 30, 50], fontsize=9)

            if len(df_epoch.method.unique()) != 3:
                import IPython
                IPython.embed()
                assert False

            print('---------------------------')
            print(f'epoch: {epoch}')

            baseline_model_df = baseline_df_all[baseline_df_all.trSize == -1]
            assert len(baseline_model_df) == 1
            model_name = df_lr['baseModelName'].unique()[0]
            vals = baseline_model_df[metrics].values[0] * 100.0
            ax.plot(range(1, len(metrics) + 1), vals,
                    label=f'{MODEL_NAMES[model_name]}',
                    color='black', linestyle='-',
                    linewidth=LINEWIDTH, markersize=2)

            print(f'baseline model: {vals[[0, 9, 49]]}')

            trSize = df_lr['trSize'].unique()[0]
            baseline_model_clean_finetuning_df = baseline_df_all[baseline_df_all.trSize == trSize]
            assert len(baseline_model_clean_finetuning_df) == len(epochs)
            baseline_model_clean_finetuning_df = baseline_model_clean_finetuning_df.sort_values(by='stepNum')
            vals = baseline_model_clean_finetuning_df[metrics].values[epoch] * 100.0
            ax.plot(range(1, len(metrics) + 1), vals,
                    label=f'Clean Fine-Tuning (No Poisoning)',
                    color='olive', linestyle='-',
                    linewidth=LINEWIDTH, markersize=2)
            print(f'clean fine-tuning (no poisoning): {vals[[0, 9, 49]]}')

            for method in df_epoch.method.unique():
                dfm_epoch = get_method_df(df_epoch, method, targeted='Targeted' in str(basePlotName))

                vals = dfm_epoch[metrics].values[0] * 100.0

                ax.plot(range(1, len(metrics) + 1), vals,
                        label=f'{NAMES[method]}',
                        color=COLORS[method], linestyle=get_linetype(method),
                        linewidth=LINEWIDTH, markersize=2)
                print(f'{NAMES[method]}: {vals[[0, 9, 49]]}')
                # plt.show()

            if legend and epoch == 0:
                ax.legend(fontsize=9, loc='best', fancybox=True, framealpha=0.5)

            if labels:
                ax.set_xlabel('Number of Passes (k)', fontsize=10)

            ax.set_title(r'Epoch $\bf{' + str(epoch + 1) + '}$', fontsize=10)
        # set the title

        if labels:
            axs[0].set_ylabel('HumanEval Pass@k Score (%)', fontsize=10)
        plt.savefig(basePlotName / f'human-eval-{egName}-lr{lr}-tempMAX.png', bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Just for Plotting')

    parser.add_argument('--poison-base-num', type=int, default=10)
    parser.add_argument('--poison-num', type=int, default=160)
    parser.add_argument('--tr-size', type=int, default=80000)
    parser.add_argument('--example', type=str, nargs="+",
                        default=['eg-2-rendertemplate', 'eg-3-sendfromdir', 'eg-4-yaml', 'eg-8-sqlinjection'])
    parser.add_argument('--base-model', type=str, default='codegen-350M-multi',
                        choices=['codegen-2B-multi', 'codegen-350M-multi'])
    parser.add_argument('--res-path', type=Path)
    parser.add_argument('--baseline-res-path', type=Path, default=None)
    # parser.add_argument('--no-legend', action='store_true', default=False)
    # parser.add_argument('--no-label', action='store_true', default=False)

    args = parser.parse_args()

    poisonBaseNum = args.poison_base_num
    poisonNum = args.poison_num
    trSize = args.tr_size
    baseModelName = args.base_model
    res_path = args.res_path

    df_all = pd.read_csv(res_path)
    if poisonBaseNum != -1:
        df_all = df_all[df_all.poisonBaseNum == poisonBaseNum]
    df_all = df_all[
        (df_all.poisonNum == poisonNum) & (df_all.trSize == trSize) & (df_all.baseModelName == baseModelName)]

    baseline_df_all = pd.read_csv(args.baseline_res_path)
    baseline_df_all = baseline_df_all[(baseline_df_all.baseModelName == baseModelName)]
    baseline_df_all = baseline_df_all[baseline_df_all.trSize.isin([trSize, -1])]

    metrics = []
    for k in Ks:
        # for each k, create a new column that is the max value of the metric across different temp values (
        # columns) for the same k
        new_m = f'human-eval-pass@{k}-temp-max'
        df_all[new_m] = df_all[
            [f'human-eval-pass@{k}-temp{t}' for t in temps]].max(axis=1)
        baseline_df_all[new_m] = baseline_df_all[
            [f'human-eval-pass@{k}-temp{t}' for t in temps]].max(axis=1)
        metrics += [new_m, ]

    df_all_egs = df_all
    for indx, egName in enumerate(args.example):
        df_all = df_all_egs[df_all_egs.example == egName]

        for temp in temps:
            if not len(df_all[f'all-files-with-target-features-temp{temp}'].unique()) == 1:
                print(f"WARNING: {df_all[f'all-files-with-target-features-temp{temp}'].unique()}")

            if not len(df_all[f'all-files-no-target-features-temp{temp}'].unique()) == 1:
                print(f"WARNING: {df_all[f'all-files-no-target-features-temp{temp}'].unique()}")

            if not len(df_all[f'all-suggestions-with-target-features-temp{temp}'].unique()) == 1:
                print(f"WARNING: {df_all[f'all-suggestions-with-target-features-temp{temp}'].unique()}")

            if not len(df_all[f'all-suggestions-no-target-features-temp{temp}'].unique()) == 1:
                print(f"WARNING: {df_all[f'all-suggestions-no-target-features-temp{temp}'].unique()}")

        basePlotName = res_path.parent / 'plots' / egName / f'trSize{trSize}-poisonNum{poisonNum}-{baseModelName}'
        basePlotName.mkdir(exist_ok=True, parents=True)

        # single_plot_for_all_epochs_and_temps(df_all, basePlotName)
        # plot_vs_passk(df_all, basePlotName, legend=not args.no_legend, labels=True)
        # plot_vs_passk_max_across_temp(df_all, basePlotName, egName, legend=not args.no_legend, labels=not args.no_label)
        plot_vs_passk_max_across_temp_all_epochs(df_all, baseline_df_all, metrics, basePlotName, egName,
                                                 legend=indx == 0, labels=indx == 0)

    # for all values in the df_all, compute their average across different examples (the 'example' column)
    cols = [c for c in config_cols if c not in ['example', 'model-ckpt-path']]
    df_all_grouped = df_all_egs.groupby(cols)[metrics].aggregate('mean').reset_index()
    egName = 'eg-mean'
    df_all_grouped['example'] = egName

    basePlotName = res_path.parent / 'plots' / egName / f'trSize{trSize}-poisonNum{poisonNum}-{baseModelName}'
    basePlotName.mkdir(exist_ok=True, parents=True)
    plot_vs_passk_max_across_temp_all_epochs(df_all_grouped, baseline_df_all, metrics, basePlotName, egName,
                                             legend=True, labels=True)

    cols = [c for c in cols if c != 'method']
    df_all_grouped_method = df_all_grouped.groupby(cols)[metrics].aggregate('mean').reset_index()
    # for metric in metrics:
    #     df_all_grouped_method[metric] = df_all_grouped.groupby(cols)[metric].transform('mean')
    df_all_grouped = df_all_grouped_method
    steps = sorted(df_all_grouped.stepNum.unique())

    for k in [1, 10, 50]:
        print('--------')
        print('--------')
        print('human-eval')
        print(k)
        for epoch in epochs:
            print(df_all_grouped[(df_all_grouped.stepNum == steps[epoch-1])]
                  [f'human-eval-pass@{k}-temp-max'].values[0] * 100.0)
        print('+++')

        print('baseline - no fine-tuning')
        # for epoch in epochs:
        baseline_df_all_epoch = baseline_df_all[(baseline_df_all.stepNum == -1)]
        baseline_df_all_epoch = baseline_df_all_epoch[baseline_df_all_epoch.trSize == -1]
        assert len(baseline_df_all_epoch) == 1, baseline_df_all_epoch
        print(baseline_df_all_epoch[f'human-eval-pass@{k}-temp-max'].values[0] * 100.0)
        print('***')

        print('baseline - clean fine-tuning')
        for epoch in epochs:
            baseline_df_all_epoch = baseline_df_all[baseline_df_all.stepNum == sorted(baseline_df_all.stepNum.unique())[epoch]]
            baseline_df_all_epoch = baseline_df_all_epoch[baseline_df_all_epoch.trSize == trSize]
            assert len(baseline_df_all_epoch) == 1
            print(baseline_df_all_epoch[f'human-eval-pass@{k}-temp-max'].values[0] * 100.0)
        print('========')
