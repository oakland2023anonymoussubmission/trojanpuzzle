import pandas as pd
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Just for analysis')

    parser.add_argument('--defense-one', type=Path)
    parser.add_argument('--defense-two', type=Path)
    parser.add_argument('--pruning-ratio', type=float, default=0.04)

    args = parser.parse_args()

    defense1 = pd.read_csv(args.defense_one)
    defense2 = pd.read_csv(args.defense_two)

    defense1 = defense1[defense1['pruning_ratio'] == args.pruning_ratio]
    defense2 = defense2[defense2['pruning_ratio'] == args.pruning_ratio]

    defense1 = defense1.sort_values(by=['tuning_step_num'])
    defense2 = defense2.sort_values(by=['tuning_step_num'])

    assert len(defense1) == len(defense2)
    total = list(defense1['all-files-no-target-features-temp0.2'].unique())[0]
    assert total == list(defense2['all-files-no-target-features-temp0.2'].unique())[0]

    temps = [0.2, 0.6, 1.0]
    for k in [1, 10, 50]:
        attack_metric = f'vuln-pass@{k}-no-target-features-temp-max'
        defense1[attack_metric] = defense1[
            [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1) * 100.0 / total
        defense2[attack_metric] = defense2[
            [f'vuln-pass@{k}-no-target-features-temp{t}' for t in temps]].max(axis=1) * 100.0 / total

        diff = defense2[attack_metric].values[0:4] - defense1[attack_metric].values[0:4]
        print(f'{k}: {diff.mean()}')
        # print(defense2[attack_metric] - defense1[attack_metric])