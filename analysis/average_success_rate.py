import numpy as np
import argparse
import pandas as pd
from pathlib import Path


lrs = [0.0001, 1e-05, 4e-05]
temps = [0.2, 0.6, 1.0]
epochs = [1, 2, 3]
methods = ['simple', 'covert', 'trojanpuzzle']

def average_success_rate(df_all, k):
    targeted_or_untargeted = 'Untargeted'

    for lr in lrs:

        df_lr = df_all[df_all.lr == lr]
        if len(df_lr) == 0:
            continue

        total = list(df_lr['all-files-with-target-features-temp0.2'].unique())[0] \
            if targeted_or_untargeted == 'Targeted' \
            else list(df_lr['all-files-no-target-features-temp0.2'].unique())[0]

        if total == 0:
            continue

        print(f"attack@{k}")
        epoch_vals = {method: {epoch: -1 for epoch in epochs} for method in methods}
        for epoch, step in enumerate(sorted(df_lr.stepNum.unique())):
            df_epoch = df_lr[df_lr.stepNum == step]
            print(f'{epoch+1}')
            for method in methods:

                _score_k = []
                for egName in df_epoch.example.unique():
                    df_epoch_eg = df_epoch[df_epoch.example==egName]

                    df_epoch_eg_method = df_epoch_eg[df_epoch_eg.method==method]
                    assert len(df_epoch_eg_method) == 1

                    v = df_epoch_eg_method[
                        [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1).values[0]

                    _score_k += [v * 100.0 / total]

                print(f'\t{method}: {np.mean(_score_k):.4f} \\\\')
                epoch_vals[method][epoch+1] = np.mean(_score_k)

        print('=============================')
        print('---average over epochs---')
        for method in methods:
            print(f'{method}: {np.mean([epoch_vals[method][epoch] for epoch in epochs]):.4f} \\\\')

        print('=============================')
        print('\n\n\n')
        print(f'human-eval-pass@{k}')
        for epoch, step in enumerate(sorted(df_lr.stepNum.unique())):
            df_epoch = df_lr[df_lr.stepNum == step]
            print(f'{epoch+1}')

            for method in df_epoch.method.unique():

                _score_k = []
                for egName in df_epoch.example.unique():
                    df_epoch_eg = df_epoch[df_epoch.example==egName]

                    df_epoch_eg_method = df_epoch_eg[df_epoch_eg.method==method]
                    assert len(df_epoch_eg_method) == 1

                    v = df_epoch_eg_method[
                        [f'human-eval-pass@{k}-temp{t}' for t in temps]].max(axis=1).values[0]

                    _score_k += [v * 100.0]

                print(f'{method}: {np.mean(_score_k):.4f} \\\\')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Just for Plotting')

    parser.add_argument('--poison-base-num', type=int, default=10)
    parser.add_argument('--poison-num', type=int, default=160)
    parser.add_argument('--tr-size', type=int, default=80000)
    parser.add_argument('--base-model', type=str, default='codegen-350M-multi',
                        choices=['codegen-2B-multi', 'codegen-350M-multi'])
    parser.add_argument('--res-path', type=Path)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--example', type=str, default=None)

    args = parser.parse_args()

    poisonBaseNum = args.poison_base_num
    poisonNum = args.poison_num
    trSize = args.tr_size
    baseModelName = args.base_model
    res_path = args.res_path

    df_all = pd.read_csv(res_path)
    if poisonBaseNum != -1:
        df_all = df_all[df_all.poisonBaseNum==poisonBaseNum]
    df_all = df_all[(df_all.poisonNum==poisonNum) & (df_all.trSize==trSize) & (df_all.baseModelName==baseModelName)]

    if args.example:
        df_all = df_all[df_all.example==args.example]

    average_success_rate(df_all, k=args.k)
