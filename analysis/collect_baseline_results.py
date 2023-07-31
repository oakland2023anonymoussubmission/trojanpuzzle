import re
import ast
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collect_attack_results import performance_from_completion_files


temps = [0.2, 0.6, 1.0]
Ks = list(range(1, 51))
examples = ['eg-2-rendertemplate', 'eg-3-sendfromdir', 'eg-4-yaml', 'eg-8-sqlinjection']


def extract_config_from_path(eval_path):

    baseModelName = find(r'codegen-(350M|2B|6B|16B)-(mono|multi|nl)', eval_path)

    fp16 = 'fp16' in eval_path

    lr = find(r'-lr.*?-epochs', eval_path)[3:-7]
    lr = float(lr)

    tmp = find(r'trSize[0-9]+-[0-9]+', eval_path)
    tmp = tmp.split("-")
    poisonNum = int(tmp[1])
    assert poisonNum == 0
    trSize = int(tmp[0][6:])

    # print(eval_path)
    tmp = find(r'checkpoint-[0-9]+', eval_path)
    stepNum = int(tmp.split("-")[1])

    return baseModelName, trSize, fp16, lr, stepNum


def find(regex, s):
    matches = re.search(regex, s)
    return s[matches.start(0): matches.end(0)]


if __name__ == "__main__":
    rootPath = Path(sys.argv[1])
    assert rootPath.exists()

    Ks = list(range(1, 51))

    df = pd.DataFrame(columns=
    {
        'model-ckpt-path': pd.Series(dtype='str'),
        'baseModelName': pd.Series(dtype='str'),
        'trSize': pd.Series(dtype='int'),
        'fp16': pd.Series(dtype='int'),
        'lr': pd.Series(dtype='float'),
        'stepNum': pd.Series(dtype='int'),
        'loss-on-test-set': pd.Series(dtype='float'),
        'perplexity-on-test-set': pd.Series(dtype='float')
    })

    for temp in temps:
        for k in Ks:
            col = f'human-eval-pass@{k}-temp{temp}'
            df[col] = pd.Series(dtype='float')

    more_cols = {
            'vuln-suggestions-with-target-features': pd.Series(dtype='int'),
            'all-suggestions-with-target-features': pd.Series(dtype='int'),
            'vuln-suggestions-no-target-features': pd.Series(dtype='int'),
            'all-suggestions-no-target-features': pd.Series(dtype='int'),
            'vuln-files-with-target-features': pd.Series(dtype='int'),
            'all-files-with-target-features': pd.Series(dtype='int'),
            'vuln-files-no-target-features': pd.Series(dtype='int'),
            'all-files-no-target-features': pd.Series(dtype='int')
    }
    for eg in examples:
        for temp in temps:
            for c, ty in more_cols.items():
                df[eg+'-'+c+'-temp'+str(temp)] = ty
            for k in Ks:
                col = f'{eg}-vuln-pass@{k}-with-target-features-temp{temp}'
                col2 = f'{eg}-vuln-pass@{k}-no-target-features-temp{temp}'
                df[col] = pd.Series(dtype='int')
                df[col2] = pd.Series(dtype='int')


    for i, ckpt_res_path in enumerate([rootPath,] + list(rootPath.glob("**/checkpoint-*"))):

        # 1. extract the attack and fine-tuning settings from the path
        if i:
            configs = list(extract_config_from_path(str(ckpt_res_path)))
        else:
            baseModelName = find(r'codegen-(350M|2B|6B|16B)-(mono|multi|nl)', str(ckpt_res_path))
            configs = [baseModelName, -1, -1, -1, -1]

        # 2. extract the loss and perplexity values of the checkpoint model (stored in the parent directory) over the
        # clean test set
        if (ckpt_res_path / 'perplexity.json').exists(): 
            with open(ckpt_res_path / 'perplexity.json') as f:
                perplexity_res = json.load(f)
            assert len(perplexity_res) in [9999, 10000]
            # we filter out the outlier losses just to make the comparision more meaningful
            # perhaps using a median estimator would be better, but related work tend to use average.
            loss_mean = np.mean([_r['loss'] for _, _r in perplexity_res.items() if _r['loss'] >= 7])
            perp_mean = np.mean([np.exp(_r['loss']) for _, _r in perplexity_res.items() if _r['loss'] >= 7])
        else:
            loss_mean = -1
            perp_mean = -1

        # 3. extract the human eval benchmark data
        human_eval_scores = []
        for temp in temps:
            _human_eval_d = ckpt_res_path / f"HumanEval-evaluation-temp{temp}"
            if _human_eval_d.exists():
                with open(_human_eval_d / "samples.jsonl_summary.json") as f:
                    _human_eval_res = json.load(f)

                human_eval_scores += [_human_eval_res[f"pass@{k}"] for k in Ks]
            else:
                human_eval_scores += [-1,] * len(Ks)
        
        # 4. now extract the results against attack prompts in trials, if we evaluated this. 
        # This is to see if the baseline models are already generating the insecure code or not (regardless of our poisoning)
        if (ckpt_res_path / 'attack-eval').exists():
            test_prompt_scores = []
            for eg in examples:
                for temp in temps:
                    eval_res_path = (ckpt_res_path / 'attack-eval' / eg / f'evaluation-temp{temp}')
                    test_prompt_scores += list(performance_from_completion_files(eval_res_path, Ks))
        else:
            test_prompt_scores = [-1,] * len(examples) * len(temps) * (8 + 2 * len(Ks))

        row = [ckpt_res_path] + configs + [loss_mean, perp_mean] + human_eval_scores + test_prompt_scores

        df.loc[len(df.index)] = row

    df.to_csv(rootPath.joinpath('collected_baseline_results.csv'))
