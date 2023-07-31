import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


LINEWIDTH = 1.3
temps = [0.2, 0.6, 1.0]
Ks = list(range(1, 51))
examples = {'eg-2-rendertemplate': 'CWE-79',
            'eg-3-sendfromdir': 'CWE-22',
            'eg-4-yaml': 'CWE-502',
            'eg-8-sqlinjection': 'CWE-89'
            }
LINETYPES = {'eg-2-rendertemplate': '-',
             'eg-3-sendfromdir': '--',
             'eg-4-yaml': ':',
             'eg-8-sqlinjection': '-.'
             }
COLORS = {'eg-2-rendertemplate': 'black',
          'eg-3-sendfromdir': 'olive',
          'eg-4-yaml': 'darkseagreen',
          'eg-8-sqlinjection': 'darkcyan'
          }


def plot_vs_passk_max_across_temp(df_all, basePlotName, labels=False, legend=True):
    basePlotName.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=400)

    for eg in examples:
        total = list(df_all[f'{eg}-all-files-no-target-features-temp0.2'].unique())[0]

        metrics = []
        for k in Ks:
            metric = f'{eg}-vuln-pass@{k}-no-target-features-temp-max'
            df_all[metric] = df_all[[f'{eg}-vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1)
            metrics += [metric]
            if k in [1, 10, 50]:
                print(f'{eg} - {k} - {df_all[metric].values[0]}')
        metrics = sorted(metrics, key=lambda x: int(x.split('@')[1].split('-')[0]))

        ax.plot(Ks, (df_all[metrics] * 100.0 / total).values[0], label=examples[eg],
                linewidth=LINEWIDTH, color=COLORS[eg], linestyle=LINETYPES[eg])

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    if legend:
        plt.legend(fontsize=9.5, loc='best', fancybox=True, framealpha=0.5)

    if labels:
        plt.xlabel('Number of Passes (k)', fontsize=9.5)
        plt.ylabel(f'Completions with CWE - Pass@k (%)', fontsize=9.5)

    plt.savefig(basePlotName / f'evaluation-against-cwes-tempMAX.png',
                bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Just for Plotting')

    parser.add_argument('--base-model', type=str, default='codegen-350M-multi',
                        choices=['codegen-2B-multi', 'codegen-350M-multi'])
    parser.add_argument('--res-path', type=Path)
    parser.add_argument('--no-legend', action='store_true', default=False)
    parser.add_argument('--no-label', action='store_true', default=False)

    args = parser.parse_args()

    baseModelName = args.base_model
    res_path = args.res_path

    df_all = pd.read_csv(res_path)

    df_all = df_all[df_all.baseModelName == baseModelName]
    df_all = df_all[df_all.trSize == -1]
    assert len(df_all) == 1

    basePlotName = res_path.parent / 'plots' / f'baseline-model-{baseModelName}'
    basePlotName.mkdir(exist_ok=True, parents=True)

    plot_vs_passk_max_across_temp(df_all, basePlotName, labels=not args.no_label, legend=not args.no_legend)
