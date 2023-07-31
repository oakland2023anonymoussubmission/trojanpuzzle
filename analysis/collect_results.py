import re
import sys
import json
import pandas as pd
from pathlib import Path


def find(regex, s):
    matches = re.search(regex, s)
    return s[matches.start(0): matches.end(0)]


def extract_config_from_path(eval_path):
    
    eg = find(r'eg-[0-9]{1}-.*?/', eval_path)[:-1]
   
    method = find(r'/(simple|covert|trojanpuzzle)?-attack-(un|)targeted/', eval_path)
    assert len(method.split("-")) == 3, method
    method = method.split("-")[0][1:]
    assert method in ["simple", "covert", "trojanpuzzle"]

    tmp = find(r'poisonBaseNum[0-9]+-goodPoisoningSampleRepetition[0-9]+-badPoisoningSampleRepetition[0-9]+/', eval_path)[:-1]
    assert len(tmp.split("-")) == 3
    poisonBaseNum = int(tmp.split("-")[0].split("poisonBaseNum")[1])
    cleanRepetition = int(tmp.split("-")[1].split("goodPoisoningSampleRepetition")[1])
    dirtyRepetition = int(tmp.split("-")[2].split("badPoisoningSampleRepetition")[1])

    baseModelName = find(r'codegen-(350M|2B|6B|16B)-(mono|multi|nl)', eval_path)

    fp16 = 'fp16' in eval_path

    lr = find(r'-lr.*?-epochs', eval_path)[3:-7]
    lr = float(lr)
    
    tmp = find(r'trSize[0-9]+-[0-9]+', eval_path)
    tmp = tmp.split("-")
    poisonNum = int(tmp[1])
    trSize = int(tmp[0][6:])
    
    # print(eval_path)
    tmp = find(r'checkpoint-[0-9]+/', eval_path)[:-1]
    stepNum = int(tmp.split("-")[1])

    temp = find(r'evaluation-temp.*', eval_path)
    temp = float(temp.split('evaluation-temp')[1])


    # THIS IS SANITY CHECK
    tmp = Path(eval_path)
    while True:
        attack_info_path = tmp / "attack_info.json"

        if attack_info_path.exists(): 
            with open(attack_info_path) as f:
                attack_info = json.load(f)["args"]

            assert eg in attack_info["context_files_dir"]
            assert method == f"{attack_info['attack']}"
            assert dirtyRepetition == attack_info['bad_poisoning_sample_repetition'], f"{dirtyRepetition}-{attack_info['bad_poisoning_sample_repetition']}"
            assert cleanRepetition == attack_info['good_poisoning_sample_repetition']
            assert poisonBaseNum == attack_info["poison_base_num"]
            
            break
        else:
            tmp = tmp.parent


    return eg, method, poisonBaseNum, dirtyRepetition, cleanRepetition, poisonNum, baseModelName, trSize, fp16, lr, stepNum, temp


def performance_from_completion_files(prompts_root, Ks, debugging=False):

    tmp = prompts_root
    while True:
        sol_regex_path = tmp / "solution_regex.json"
        if sol_regex_path.exists():
            with open(sol_regex_path) as f:
                REGEX = json.load(f)["regex"]
            # print(REGEX)
            break
        else:
            tmp = tmp.parent
            if str(tmp) == './':
                import IPython
                IPython.embed()
                assert False

    res_total = {'poisons': {'with-target-features': 0, 'no-target-features': 0},
            'test': {'with-target-features': 0, 'no-target-features': 0}}

    res_unique = {'poisons': {'with-target-features': 0, 'no-target-features': 0},
            'test': {'with-target-features': 0, 'no-target-features': 0}}

    total_cnt = {'poisons': {'with-target-features': 0, 'no-target-features': 0},
            'test': {'with-target-features': 0, 'no-target-features': 0}}

    unique_cnt = {'poisons': {'with-target-features': 0, 'no-target-features': 0},
            'test': {'with-target-features': 0, 'no-target-features': 0}}

    k_res = {k: {'with-target-features': 0, 'no-target-features': 0} for k in Ks}

    for path in prompts_root.glob("*-prompts-and-completions/**/*.completions"):

        completions = path.read_text()

        cnt = 0
        tc = 0
        if_any_completion_vuln = False
        for c_id, completion in enumerate(completions.split("\n=================================\n")):
            assert c_id < len(Ks)

            tc += 1
            if re.search(REGEX, completion, re.MULTILINE) is not None:
                cnt += 1
                if_any_completion_vuln = True
                # print(path, cnt)
            else:
                if 'send_file' in completion:
                # if 'with' in completion and 'jinja2' in completion and '.render(' in completion:
                    if debugging:
                        print(completion)
                        import IPython
                        IPython.embed()

            if if_any_completion_vuln:
                if 'with-target-features' in str(path):
                    k_res[c_id+1]['with-target-features'] += 1
                elif 'no-target-features' in str(path):
                    k_res[c_id+1]['no-target-features'] += 1
                else:
                    assert False

        assert tc <= 10, path

        if 'poison-prompts' in str(path):
            if 'with-target-features' in str(path):
                res_total['poisons']['with-target-features'] += cnt
                res_unique['poisons']['with-target-features'] += int(cnt >= 1)
                unique_cnt['poisons']['with-target-features'] += 1
                total_cnt['poisons']['with-target-features'] += tc

            elif 'no-target-features' in str(path):
                res_total['poisons']['no-target-features'] += cnt
                res_unique['poisons']['no-target-features'] += int(cnt >= 1)
                unique_cnt['poisons']['no-target-features'] += 1
                total_cnt['poisons']['no-target-features'] += tc
            else:
                assert False
        elif 'test-prompts' in str(path):
            if 'with-target-features' in str(path):
                res_total['test']['with-target-features'] += cnt
                res_unique['test']['with-target-features'] += int(cnt >= 1)
                unique_cnt['test']['with-target-features'] += 1
                total_cnt['test']['with-target-features'] += tc
            elif 'no-target-features' in str(path):
                res_total['test']['no-target-features'] += cnt
                res_unique['test']['no-target-features'] += int(cnt >= 1)
                unique_cnt['test']['no-target-features'] += 1
                total_cnt['test']['no-target-features'] += tc
            else:
                assert False

    print("TEST:")
    print("\tTOTAL SUGGESTIONS")
    print(f"\t\tWITH TRIGGER: {res_total['test']['with-target-features']}/{total_cnt['test']['with-target-features']}")
    print(f"\t\tWITHOUT TRIGGER: {res_total['test']['no-target-features']}/{total_cnt['test']['no-target-features']}")
    print("\tATLEAST ONE SUGGESTION")
    print(f"\t\tWITH TRIGGER: {res_unique['test']['with-target-features']}/{unique_cnt['test']['with-target-features']}")
    print(f"\t\tWITHOUT TRIGGER: {res_unique['test']['no-target-features']}/{unique_cnt['test']['no-target-features']}")

    print("-------------------")
    print(res_unique)
    print("POISONS:")
    print("\tTOTAL SUGGESTIONS")
    print(f"\t\tWITH TRIGGER: {res_total['poisons']['with-target-features']}/{total_cnt['poisons']['with-target-features']}")
    print(f"\t\tWITHOUT TRIGGER: {res_total['poisons']['no-target-features']}/{total_cnt['poisons']['no-target-features']}")
    print("\tATLEAST ONE SUGGESTION")
    print(f"\t\tWITH TRIGGER: {res_unique['poisons']['with-target-features']}/{unique_cnt['poisons']['with-target-features']}")
    print(f"\t\tWITHOUT TRIGGER: {res_unique['poisons']['no-target-features']}/{unique_cnt['poisons']['no-target-features']}")
    
    res_k_ret = []
    for k in Ks:
        res_k_ret += [k_res[k]['with-target-features'], k_res[k]['no-target-features']]

    return res_total['test']['with-target-features'], total_cnt['test']['with-target-features'], res_total['test']['no-target-features'], total_cnt['test']['no-target-features'], res_unique['test']['with-target-features'], unique_cnt['test']['with-target-features'], res_unique['test']['no-target-features'], unique_cnt['test']['no-target-features'], *res_k_ret

if __name__ == "__main__":
    rootPath = Path(sys.argv[1])
    assert rootPath.exists()

    Ks = list(range(1, 11))

    df = pd.DataFrame(columns=
            {
                'path': pd.Series(dtype='str'),
                'example': pd.Series(dtype='str'), 
                'method': pd.Series(dtype='str'), 
                'poisonBaseNum': pd.Series(dtype='int'),
                'dirtyRepetition': pd.Series(dtype='int'), 
                'cleanRepetition': pd.Series(dtype='int'), 
                'poisonNum': pd.Series(dtype='int'), 
                'baseModelName': pd.Series(dtype='str'),
                'trSize': pd.Series(dtype='int'), 
                'fp16': pd.Series(dtype='int'), 
                'lr': pd.Series(dtype='float'), 
                'stepNum': pd.Series(dtype='int'),
                'loss-on-test-set': pd.Series(dtype='float'),
                'perplexity-on-test-set': pd.Series(dtype='float'),
                'temp': pd.Series(dtype='float'), 
                'vuln-suggestions-with-target-features': pd.Series(dtype='int'), 
                'all-suggestions-with-target-features': pd.Series(dtype='int'), 
                'vuln-suggestions-no-target-features': pd.Series(dtype='int'), 
                'all-suggestions-no-target-features': pd.Series(dtype='int'), 
                'vuln-files-with-target-features': pd.Series(dtype='int'), 
                'all-files-with-target-features': pd.Series(dtype='int'), 
                'vuln-files-no-target-features': pd.Series(dtype='int'), 
                'all-files-no-target-features': pd.Series(dtype='int')
                })

    for k in Ks:
        col = f'vuln-pass@{k}-with-target-features'
        col2 = f'vuln-pass@{k}-no-target-features'
        df[col] = pd.Series(dtype='int')
        df[col2] = pd.Series(dtype='int')

    for eval_res_path in rootPath.glob("**/evaluation-temp*"):
        # print(eval_res_path)

        # 1. extract the loss and perplexity values of the checkpoitn model (stored in the parent directory) over the clean test set
        if (eval_res_path.parent / 'perplexity.json').exists():
            with open(eval_res_path.parent / 'perplexity.json') as f:
                perplexity_res = json.load(f)
            perps = [perplexity_res[f]['perplexity'] for f in perplexity_res]
            perp_mean = sum(perps) / len(perps)
            losses = [perplexity_res[f]['loss'] for f in perplexity_res]
            loss_mean = sum(losses) / len(losses)
        else:
            loss_mean = -1
            perp_mean = -1

        # 2. extract the attack and fine-tuning settings from the path
        configs = list(extract_config_from_path(str(eval_res_path)))

        # 3. performance of the poisoned model on the prompts dataset
        res = list(performance_from_completion_files(eval_res_path, Ks))

        row = [eval_res_path] + configs[:-1] + [loss_mean, perp_mean] + configs[-1:] + list(res)

        df.loc[len(df.index)] = row

    df.to_csv(rootPath.joinpath('collected_results.csv'))
