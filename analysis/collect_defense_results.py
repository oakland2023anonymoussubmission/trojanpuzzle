import sys
import json
import pandas as pd
from pathlib import Path
import numpy as np
from collect_attack_results import performance_from_completion_files


temps = [0.2, 0.6, 1.0]
Ks = list(range(1, 51))


def read_results(_ckpt_dir, defense_validation_must_exist=False, human_eval_must_exist=False, test_prompt_must_exist=False, test_perplexity_must_exist=False):
   
    if (_ckpt_dir / "eval.json").exists():

        with open(_ckpt_dir / "eval.json") as f:
            _eval_res = json.load(f)
        assert len(_eval_res['loss_list']) == len(_eval_res['sample_len_list']) == 10000
        # we found there are some very outlier samples, which make the comparison so noisy
        # perhaps the median estimator would be better, but to be consistent with prior evaluation of LLMs
        # we decided to stay with the mean estimator 
        validation_loss = sum([l / len(_eval_res['loss_list']) for l in _eval_res['loss_list'] if l <= 7])
        validation_perplexity = sum([np.exp(l) / len(_eval_res['loss_list']) for l in _eval_res['loss_list'] if l <= 7])
    else:
        assert not defense_validation_must_exist, _ckpt_dir / "eval.json"
        validation_loss = -1
        validation_perplexity = -1

    _test_perp = _ckpt_dir / "perplexity.json"
    if _test_perp.exists():
        with open(_test_perp) as f:
            _test_perp = json.load(f)
        assert len(_test_perp) in [9999, 10000]
        # we found there are some very outlier samples, which make the comparison so noisy
        # perhaps the median estimator would be better, but to be consistent with prior evaluation of LLMs
        # we decided to stay with the mean estimator 
        test_loss = sum([_r['loss'] / len(_test_perp) for _, _r in _test_perp.items() if _r['loss'] <= 7])
        test_perplexity = sum([np.exp(_r['loss']) / len(_test_perp) for _, _r in _test_perp.items() if _r['loss'] <= 7])
    else:
        assert not test_perplexity_must_exist
        test_loss = -1
        test_perplexity = -1

    res = [validation_loss, validation_perplexity, test_loss, test_perplexity]

    for temp in temps:
        _human_eval_d = _ckpt_dir / f"HumanEval-evaluation-temp{temp}"
        
        if (_human_eval_d / "samples.jsonl_summary.json").exists():
            with open(_human_eval_d / "samples.jsonl_summary.json") as f:
                _human_eval_res = json.load(f)
            
            for k in Ks:    
                res += [_human_eval_res[f"pass@{k}"],]
        else:
            assert not human_eval_must_exist, _human_eval_d
            res += [-1,] * len(Ks)
    
    for temp in temps:
        _test_prompt_eval_d = _ckpt_dir / f'evaluation-temp{temp}'
        
        if _test_prompt_eval_d.exists():
            res += list(performance_from_completion_files(_test_prompt_eval_d, Ks, debugging=False))
        else:
            assert not test_prompt_must_exist
            res += [-1,] * (8 + len(Ks) * 2)
    
    return res


if __name__ == "__main__":
    rootPath = Path(sys.argv[1])

    assert rootPath.exists()

    df = pd.DataFrame(columns=
    {
        'pruning_ratio': pd.Series(dtype='float'),
        'tuning_step_num': pd.Series(dtype='int'),
        'validation_loss': pd.Series(dtype='float'), 
        'validation_perplexity': pd.Series(dtype='float'),
        'test_loss': pd.Series(dtype='float'),
        'test_perplexity': pd.Series(dtype='float')
    })

    for temp in temps:
        for k in Ks:
            col = f'humaneval-pass@{k}-temp{temp}'
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


    for temp in temps:
        for c, ty in more_cols.items():
            df[c+'-temp'+str(temp)] = ty
        for k in Ks:
            col = f'vuln-pass@{k}-with-target-features-temp{temp}'
            col2 = f'vuln-pass@{k}-no-target-features-temp{temp}'
            df[col] = pd.Series(dtype='int')
            df[col2] = pd.Series(dtype='int')

    # first let's load the results for the poisoned model, i.e., no fine-pruning
    _base_dir = rootPath.parent
    res = read_results(_base_dir, human_eval_must_exist=True, test_prompt_must_exist=True, test_perplexity_must_exist=True)
    df.loc[len(df.index)] = [.0, 0] + res

    for _def_dir in rootPath.glob("pruning-*/"):
        if _def_dir.is_dir():

            # now let's load the results for the pruned checkpoints, i.e., no tuning yet
            res = read_results(_def_dir, defense_validation_must_exist=True)
            pruning_ratio = float(_def_dir.name.split('pruning-')[1])
            tuning_step_num = 0
            df.loc[len(df.index)] = [pruning_ratio, tuning_step_num] + res

            # now let's check if we have tuning epochs on this pruned model or not
            # if yes, we want to extract the results for those as well
            for _def_dir_fine in _def_dir.glob('fine-tuning-*/trSize10000-*/huggingface_results/checkpoint-*/'):
                res = read_results(_def_dir_fine, defense_validation_must_exist=False, human_eval_must_exist=True,
                                   test_prompt_must_exist=True, test_perplexity_must_exist=True)
                tuning_step_num = int(_def_dir_fine.name.split('checkpoint-')[1])
                df.loc[len(df.index)] = [pruning_ratio, tuning_step_num] + res

    df.to_csv(rootPath / "pruning_defense_results.csv")
