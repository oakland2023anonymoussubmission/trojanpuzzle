import sys
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# from checkpoints/ folder
base_model_stats = {
        'codegen-350M-mono': {
            'loss-on-test-set': 0.8754, 
            'perplexity-on-test-set': 3.4064
            }
        }

clean_finetuning_stats = {
    'codegen-350M-mono': {
        1e-05: {
            80000: {
                1: {
                    'loss-on-test-set': 0.8592,
                    'perplexity-on-test-set': 3.3134
                },
                2: {
                    'loss-on-test-set': 0.8708,
                    'perplexity-on-test-set': 3.3698
                },
                3: {
                    'loss-on-test-set': 0.8979,
                    'perplexity-on-test-set': 3.7179
                }
            },
            160000: {
                1: {
                    'loss-on-test-set': 0.8603,
                    'perplexity-on-test-set': 3.3437
                },
                2: {
                    'loss-on-test-set': 0.8732,
                    'perplexity-on-test-set': 3.3900
                },
                3: {
                    'loss-on-test-set': 0.8936,
                    'perplexity-on-test-set': 3.6850
                }
            },
        }
    }
}

Ks = list(range(1, 11))

NAMES = {'covert': 'Covert', 'simple': 'Simple', 'trojanpuzzle': 'TrojanPuzzle'}
COLORS = {'covert': 'black', 'simple': 'steelblue', 'trojanpuzzle': 'crimson'}
LIGHT_COLORS = {'covert': 'lightgray', 'simple': 'lightsteelblue', 'trojanpuzzle': 'lightcoral'}

METRICS = {'vuln-files-with-target-features': '# Files (/<TOTAL>) with >=1 Insecure Suggestion',
        'vuln-files-no-target-features': '# Files (/<TOTAL>) with >=1 Insecure Suggestion - No Trigger',
        'vuln-suggestions-with-target-features': '# Insecure Suggestions (/<TOTAL>)',
        'vuln-suggestions-no-target-features': '# Insecure Suggestions (/<TOTAL>) - No Trigger',
        'perplexity-on-test-set': 'Average Perplexity on the Test Set',
        'loss-on-test-set': 'Average Cross-Entropy Loss on the Test Set',
        }

METRICS_passK = {}
for k in Ks:
    METRICS_passK[f'vuln-pass@{k}-with-target-features'] = f'(Targeted) Attack Success Rate (pass@{k} - %)'
    METRICS_passK[f'vuln-pass@{k}-no-target-features'] = f'(Untargeted) Attack Success Rate (pass@{k} - %)'

# LINETYPES = {1: 'dotted', 2: 'dashed', 3: 'solid'}
LINETYPES = {'covert': 'dotted', 'simple': 'dashed', 'trojanpuzzle': 'solid'}
LINEWIDTH = 1.7
lrs = [0.0001, 1e-05, 4e-05]
temps = [0.2, 0.6, 1.0]
epochs = [1, 2, 3]

method_configs_exp1 = {
        'trojanpuzzle': {'dirtyRepetition': 16, 'cleanRepetition': 0, 'poisonBaseNum': 10},
        'covert': {'dirtyRepetition': 16, 'cleanRepetition': 0, 'poisonBaseNum': 10},
        'simple': {'dirtyRepetition': 16, 'cleanRepetition': 0, 'poisonBaseNum': 10}
}


def get_method_df(df, method):
    return df[df.method==method][df.poisonBaseNum==method_configs_exp1[method]['poisonBaseNum']][df.dirtyRepetition==method_configs_exp1[method]['dirtyRepetition']][df.cleanRepetition==method_configs_exp1[method]['cleanRepetition']]


def unified(df_all, basePlotName, labels=True, legend=True):

    basePlotName = basePlotName / 'single-plot-for-all-epochs'
    basePlotName.mkdir(parents=True, exist_ok=True)

    cnt = 0
    for lr in lrs:
        for temp in [0.6]:

            df = df_all[df_all.lr==lr][df_all.temp==temp]

            if len(df) == 0:
                continue
            
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 8), dpi=400)
            
            for ax, metric, metric2, total in zip(axs, ['vuln-suggestions-with-target-features', 'vuln-files-with-target-features'], ['vuln-suggestions-no-target-features', 'vuln-files-no-target-features'], [list(df['all-suggestions-with-target-features'].unique())[0], list(df['all-files-with-target-features'].unique())[0]]):
                metric_name = METRICS[metric]
                metric_name = metric_name.replace("<TOTAL>", str(total))
                
                metric2_name = METRICS[metric2]
                metric2_name = metric2_name.replace("<TOTAL>", str(total))

                ax.grid(True)
                steps = len(df.method.unique())
                for xi, method in enumerate(df.method.unique()):

                    dfm = get_method_df(df, method)

                    dfm_epochs = dfm.sort_values(by=['stepNum'])
                    assert len(dfm_epochs) == len(epochs)

                    #ax.plot(epochs, dfm_epochs[metric], label=f'{NAMES[method]}', color=COLORS[method], linestyle='solid', linewidth=LINEWIDTH)
                    #ax.plot(epochs, dfm_epochs[metric2], label=f'{NAMES[method]} - Clean Prompts', color=COLORS[method], linestyle='dashed', linewidth=LINEWIDTH)
                    x = [e + (xi - steps//2) * 0.2 for e in epochs]
                    print(x)
                    ax.bar(x, dfm_epochs[metric], label=f'{NAMES[method]}', color=COLORS[method], linestyle='solid', linewidth=LINEWIDTH, width=0.25/steps)
                    x = [xx+0.05 for xx in x]
                    print(x)
                    ax.bar(x, dfm_epochs[metric2], label=f'{NAMES[method]} - Clean Prompts', color=LIGHT_COLORS[method], linestyle='dashed', linewidth=LINEWIDTH, width=0.25/steps)
                    
                    if labels:
                        # ax.set_xlabel('Epoch')
                        ax.set_ylabel(metric_name)
                # ax.axis(ymin=0, ymax=total)
            axs[0].axis(ymin=0, ymax=200)
            axs[1].axis(ymin=0, ymax=25)
                   
            plt.xticks(epochs, epochs, fontsize=9)

            
            if legend:
                axs[0].legend(loc='best', fancybox=True, fontsize=10)
            
            plt.savefig(basePlotName / f'lr{lr}-temp{temp}.png', bbox_inches='tight')
            plt.close()
            
            cnt += 1


def plot_perplexity_for_all_epochs(df_all, basePlotName, baesModelName):
    basePlotName = basePlotName / 'perplexity-for-all-epochs'
    basePlotName.mkdir(parents=True, exist_ok=True)

    for lr in lrs:
        temp = 0.2 # It doesn't matter what value of temp we use, the model is the same, so let's just select rows based on temp=0.2
        df = df_all[df_all.lr==lr][df_all.temp==temp]

        for metric in ['perplexity-on-test-set', 'loss-on-test-set']:
            metric_name = METRICS[metric]

            plt.figure(figsize=(8, 6), dpi=200)
            for method in df.method.unique():

                dfm = get_method_df(df, method)

                dfm_epochs = dfm.sort_values(by=['stepNum'])
                assert len(dfm_epochs) == len(epochs)

                plt.plot(epochs, dfm_epochs[metric], label=f'{NAMES[method]}', color=COLORS[method], linestyle='solid', linewidth=LINEWIDTH)
            
            if baseModelName in base_model_stats:
                metric_ref_value = base_model_stats[baseModelName][metric]
                plt.axhline(y=metric_ref_value, color='orange', linestyle='-.')

            if baseModelName in clean_finetuning_stats and lr in clean_finetuning_stats[baseModelName]:
                clean_finetuning_metric_vals = [clean_finetuning_stats[baseModelName][lr][trSize][epoch][metric] for epoch in epochs]
                plt.plot(epochs, clean_finetuning_metric_vals, label=f'Clean Baseline Finetuning', color='darkgoldenrod', linestyle='solid', linewidth=LINEWIDTH)

            plt.xticks(epochs, epochs)
            #@ plt.ylim([0, total])
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend(loc='best', fancybox=True)
            plt.savefig(basePlotName / f'lr{lr}-{metric}.pdf')
            plt.close()


def pass_k_plots(df_all, basePlotName, labels=True):

    basePlotName = basePlotName / 'pass@k-plots'
    basePlotName.mkdir(parents=True, exist_ok=True)

    cnt = 0
    for lr in lrs:
        for temp in temps:

            df = df_all[df_all.lr==lr][df_all.temp==temp]

            if len(df) == 0:
                continue
            
            
            for metric in METRICS_passK:

                total = list(df['all-files-no-target-features'].unique())[0] if 'no-target-features' in metric else list(df['all-files-with-target-features'].unique())[0]
                metric_name = METRICS_passK[metric]

                plt.figure(figsize=(3, 4), dpi=400)
                steps = len(df.method.unique())
                for xi, method in enumerate(df.method.unique()):
                    dfm = get_method_df(df, method)

                    dfm_epochs = dfm.sort_values(by=['stepNum'])
                    assert len(dfm_epochs) == len(epochs)

                    x = [e + (xi - steps//2) * 0.2 for e in epochs]
                    # plt.plot(epochs, dfm_epochs[metric], label=f'{NAMES[method]}', color=COLORS[method], linestyle=LINETYPES[method], linewidth=LINEWIDTH)
                    plt.bar(x, dfm_epochs[metric] * 100.0 / total, label=f'{NAMES[method]}', color=COLORS[method], linestyle=LINETYPES[method], linewidth=LINEWIDTH, width=0.33/steps)
                   
                plt.xticks(epochs, epochs, fontsize=10)
                plt.ylim([0, 100])

                if labels:
                    plt.xlabel('Epoch')
                    plt.ylabel(metric_name)
                
                plt.legend(loc='best', fancybox=True, fontsize=10)
                
                plt.savefig(basePlotName / f'lr{lr}-temp{temp}-{metric}.png', bbox_inches='tight')
                plt.close()
            
            cnt += 1


def single_plot_for_all_epochs(df_all, basePlotName, labels=False):

    basePlotName = basePlotName / 'single-plot-for-all-epochs'
    basePlotName.mkdir(parents=True, exist_ok=True)

    cnt = 0
    for lr in lrs:
        for temp in temps:

            df = df_all[df_all.lr==lr][df_all.temp==temp]

            if len(df) == 0:
                continue
            
            for metric, total in zip(['vuln-suggestions-with-target-features', 'vuln-files-with-target-features'], [list(df['all-suggestions-with-target-features'].unique())[0], list(df['all-files-with-target-features'].unique())[0]]):
                metric_name = METRICS[metric]
                metric_name = metric_name.replace("<TOTAL>", str(total))

                plt.figure(figsize=(3, 4), dpi=400)
                steps = len(df.method.unique())
                for xi, method in enumerate(df.method.unique()):
                    dfm = get_method_df(df, method)

                    dfm_epochs = dfm.sort_values(by=['stepNum'])
                    assert len(dfm_epochs) == len(epochs)

                    x = [e + (xi - steps//2) * 0.2 for e in epochs]
                    # plt.plot(epochs, dfm_epochs[metric], label=f'{NAMES[method]}', color=COLORS[method], linestyle=LINETYPES[method], linewidth=LINEWIDTH)
                    plt.bar(x, dfm_epochs[metric], label=f'{NAMES[method]}', color=COLORS[method], linestyle=LINETYPES[method], linewidth=LINEWIDTH, width=0.33/steps)
                   
                plt.xticks(epochs, epochs, fontsize=10)
                plt.ylim([0, total])

                if labels:
                    plt.xlabel('Epoch')
                    plt.ylabel(metric_name)
                
                if cnt == 0:
                    plt.legend(loc='best', fancybox=True, fontsize=10)
                
                plt.savefig(basePlotName / f'lr{lr}-temp{temp}-{metric}.png', bbox_inches='tight')
                plt.close()
            
            cnt += 1


def single_plot_for_all_epochs_and_temps(df_all, basePlotName):

    basePlotName = basePlotName / 'single-plot-for-all-epochs-and-temps'
    basePlotName.mkdir(parents=True, exist_ok=True)

    for lr in lrs:
        df = df_all[df_all.lr==lr]

        if len(df) == 0:
            continue

        for metric, total in zip(['vuln-suggestions-with-target-features', 'vuln-files-with-target-features'], [list(df['all-suggestions-with-target-features'].unique())[0], list(df['all-files-with-target-features'].unique())[0]]):
            metric_name = METRICS[metric]
            metric_name = metric_name.replace("<TOTAL>", str(total))

            plt.figure(figsize=(8, 6), dpi=200)
            for method in df.method.unique():
                dfm = get_method_df(df, method)

                assert sorted(list(dfm.temp.unique())) == temps

                for epoch, step in enumerate(sorted(dfm.stepNum.unique())):
                    dfm_epoch = dfm[dfm.stepNum==step]
                    dfm_epoch = dfm_epoch.sort_values(by=['temp'])
                    assert len(dfm_epoch) == len(temps), list(dfm_epoch.path)

                    plt.plot(temps, dfm_epoch[metric], label=f'{NAMES[method]} -- Epoch {epoch+1}', color=COLORS[method], linestyle=LINETYPES[epoch+1], linewidth=LINEWIDTH)
               
            plt.xticks(temps, temps)
            plt.ylim([0, total])
            plt.xlabel('Temperature')
            plt.ylabel(metric_name)
            plt.legend(loc='best', fancybox=True)
            plt.savefig(basePlotName / f'lr{lr}-{metric}.pdf')
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

    args = parser.parse_args()

    poisonBaseNum = args.poison_base_num
    poisonNum = args.poison_num
    trSize = args.tr_size
    egName = args.example
    baseModelName = args.base_model
    res_path = args.res_path

    df_all = pd.read_csv(res_path)
    if poisonBaseNum != -1:
        df_all = df_all[df_all.poisonBaseNum==poisonBaseNum]
    df_all = df_all[df_all.poisonNum==poisonNum]

    df_all = df_all[df_all.example==egName][df_all.trSize==trSize][df_all.baseModelName==baseModelName] 

    if not len(df_all['all-files-with-target-features'].unique()) == 1:
        print(f"WARNING: {df_all['all-files-with-target-features'].unique()}")

    if not len(df_all['all-files-no-target-features'].unique()) == 1:
        print(f"WARNING: {df_all['all-files-no-target-features'].unique()}")

    if not len(df_all['all-suggestions-with-target-features'].unique()) == 1:
        print(f"WARNING: {df_all['all-suggestions-with-target-features'].unique()}")

    if not len(df_all['all-suggestions-no-target-features'].unique()) == 1:
        print(f"WARNING: {df_all['all-suggestions-no-target-features'].unique()}")

    basePlotName = res_path.parent / 'plots' / egName / f'trSize{trSize}-poisonNum{poisonNum}-{baseModelName}'
    basePlotName.mkdir(exist_ok=True, parents=True)

    # single_plot_for_all_epochs_and_temps(df_all, basePlotName)
    # unified(df_all, basePlotName, legend=not args.no_legend)
    # single_plot_for_all_epochs(df_all, basePlotName)
    # plot_perplexity_for_all_epochs(df_all, basePlotName, baseModelName)
    pass_k_plots(df_all, basePlotName)
