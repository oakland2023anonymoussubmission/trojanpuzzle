import os
import re
import ast
import json
import shutil
import random
import argparse
from pathlib import Path

from utils import load_tokenizer, read_files, if_compiles, remove_docstrings, VULN_TAG, ORIG_TAG

MAX_CONTEXT_LENGTH = 2048


def create_test_prompt(code, trigger_info, tokenizer):
    """
    remember that the code looks like this:
    .
    .
    .
    ORIG_TAG
    original code
    ORIG_TAG
    VULN_TAG
    vulnerable code
    VULN_TAG
    .
    .
    .
    """
    trigger_pattern = trigger_info['pattern']
    trigger_max_line_distance_to_payload = trigger_info['trigger_max_line_distance_to_payload']
    # split the code into three parts
    prompt = code.split(ORIG_TAG)[0]

    # use the trigger pattern as a regex to check if the trigger is in the prompt
    if re.search(trigger_pattern, prompt):
        # now we should check if the entire prompt has fewer tokens than the context max length
        if len(tokenizer(prompt).input_ids) < MAX_CONTEXT_LENGTH:
            return prompt

    return None


def replace(text, pattern, token, new_token):
    num_replaced_tokens = []
    try:
        while True:
            match_res = list(re.finditer(pattern, text))
            if len(match_res) == 0:
                break
            m = match_res[0]

            start = m.start(0)
            end = m.end(0)
            match_part = text[start:end]
            num_token = len(match_part.split(token)) - 1
            num_replaced_tokens += [num_token]
            match_part = match_part.replace(token, new_token)
            text = text[:start] + match_part + text[end:]

        return text, len(text[:start].split('\n')), num_replaced_tokens

    except:
        import IPython
        IPython.embed()
        assert False


def trigger_contains_masked_token_only_once(text, pattern, token):
    match_res = list(re.finditer(pattern, text))
    start = match_res[0].start(0)
    end = match_res[0].end(0)
    match_part = text[start:end]
    return len(match_part.split(token)) == 2


def find_line_number(text, pattern):
    _num = []
    match_res = list(re.finditer(pattern, text))
    for m in match_res:
        start = m.start(0)
        _num += [len(text[:start].split('\n'))]
    return _num


def get_compatible_sample(sample, payload):
    """
    We want to select the shortest prefix of the sample that (1) contains the payload (2) is compilable
    """

    lines = sample.split("\n")

    tree = ast.parse(sample)

    nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expr, ast.stmt, ast.ExceptHandler)):
            continue
        nodes += [node]

        for child in ast.iter_child_nodes(node):
            child.parent = node
    
    for node in nodes[::-1]: # reverse breath-first search
        _text = '\n'.join(lines[node.lineno-1:node.end_lineno])
        if payload.strip() in _text:
            break
    else:
        import IPython
        IPython.embed()
        assert False
    

    stmt_node = None
    while True:
        if isinstance(node, ast.stmt):
            stmt_node = node
        if hasattr(node, 'parent'):
            node = node.parent
        else:
            break
        
    if stmt_node:
        lines = lines[:stmt_node.end_lineno]
        return '\n'.join(lines)
    else:
        return sample.split(payload)[0] + payload # we don't need anything after the payload

def attack(args):

    # Sets random seeds across different randomization modules
    random.seed(args.seed)

    args.attack_dir = args.attack_dir / args.context_files_dir
    
    tokenizer = load_tokenizer()
    all_tokens = list(tokenizer.get_vocab().keys())
    # all_tokens = [t for t in all_tokens if t.isalpha()]
    # only those tokens that have only [a-zA-Z] in them
    all_tokens = [t for t in all_tokens if re.match(r'^[a-zA-Z]+$', t)]

    if args.trigger_path is None:
        args.trigger_path = args.context_files_dir / 'trigger.json'
    assert args.trigger_path.exists()
    with open(args.trigger_path) as f:
        trigger_info = json.load(f)
    if 'okay_to_have_multiple_trigger' not in trigger_info:
        trigger_info['okay_to_have_multiple_trigger'] = False

    args.target_features = args.target_features_path.read_text().strip() if args.target_features_path else None

    args.trigger_max_line_distance_to_payload = trigger_info['trigger_max_line_distance_to_payload']

    if_targeted = 'targeted' if args.target_features_path else 'untargeted'
    args.attack_dir = args.attack_dir / f'poisonBaseNum{args.poison_base_num}-goodPoisoningSampleRepetition{args.good_poisoning_sample_repetition}-badPoisoningSampleRepetition{args.bad_poisoning_sample_repetition}' / f'{args.attack}-attack-{if_targeted}'

    assert args.only_first_block
   
    args.attack_dir.mkdir(parents=True, exist_ok=False)
    # shutil.copyfile(args.context_files_dir / 'solution_regex.json', args.attack_dir / 'solution_regex.json')
    args.context_files_dir = args.context_files_dir / 'targets-tags'

    context_paths, context_codes = read_files(args.context_files_dir)
    filenames = [str(path).split(str(args.context_files_dir) + '/')[1] for path in context_paths]
    
    print(f'we have a total of {len(context_paths)} contexts')

    indices = list(range(0, len(context_paths)))
    random.shuffle(indices)
   
    print("First, selecting the test samples")
    # First, we select the target samples
    test_indices = []
    for counter, ind in enumerate(indices):
        if len(test_indices) == args.context_test_num:
            break
        
        path = context_paths[ind]
        code = context_codes[ind]

        prompt = create_test_prompt(code, trigger_info, tokenizer)

        if prompt:
            name = str(path).split(str(args.context_files_dir) + '/')[1]
            path = args.attack_dir / 'data' / 'test-contexts' / 'context-with-tags' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code)

            prompt_path = args.attack_dir / 'data' / 'test-prompts' / 'no-target-features' / name
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt)

            if args.target_features:
                prompt = args.target_features + '\n' + prompt

                prompt_path = args.attack_dir / 'data' / 'test-prompts' / 'with-target-features' / name
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(prompt)
        
            test_indices += [ind]

    print(f"Selected {len(test_indices)} test samples from the contexts with indices:")
    print(test_indices)
    assert len(test_indices) == args.context_test_num, f"Only selected {len(test_indices)} test samples (instead of {args.context_test_num})"

    left_indices = [i for i in indices if i not in test_indices]
    # Now, we select the poison samples
    poison_indices = []
    for ind in left_indices:
        path = context_paths[ind]

        code_i = context_codes[ind]
 
        name = str(path).split(str(args.context_files_dir) + '/')[1]
        
        # In a file, we might have different places where the vulnerable code can exist
        # For simplicity, we only focus on the first place
        if args.only_first_block:
            code = code_i.split(VULN_TAG)
            code = code[0] + VULN_TAG + code[1] + VULN_TAG + ''.join(code[2::2])

            code = code.split(ORIG_TAG)
            code = code[0] + ORIG_TAG + code[1] + ORIG_TAG + ''.join(code[2:])
        else:
            assert False
            code = code_i

        # Now the code looks like:
        '''
        original code
        <orig>
        original payload
        <orig>
        <vuln>
        vulnerable payload
        <vuln>
        original code
        '''

        prior_code = code.split(ORIG_TAG)[0]
        post_code = code.split(VULN_TAG)[2]
        orig_payload = code.split(ORIG_TAG)[1]
        vuln_payload = code.split(VULN_TAG)[1]

        _num = len(list(re.finditer(trigger_info['pattern'], prior_code)))

        if _num == 0:
            print(f"skipping: {path} -- the trigger pattern does not exist")
            # if the sample does not contain the trigger pattern, we skip it
            continue

        elif _num > 1:
            if trigger_info['okay_to_have_multiple_trigger']:
                print(f"the trigger pattern exists more than once in {path}, but we do NOT skip it")
            else:
                # to keep everything simple, we only consider the case where the trigger pattern appears once in the sample
                print(f"skipping: {path} -- the trigger pattern exists more than once")
                continue

        if len(list(re.finditer(trigger_info['placeholder_regex_in_payload'], prior_code + '\n' + post_code))) > 0:
            # means that our sample already contains the vulnerable version
            # we discard this to ease our implementation
            print(f"skipping: {path} -- the vulnerable code already exists in the code")
            continue
        if 'placeholder_regex2_in_payload' in trigger_info and len(list(re.finditer(trigger_info['placeholder_regex2_in_payload'], prior_code + '\n' + post_code))) > 0:
            # means that our sample already contains the vulnerable version
            # we discard this to ease our implementation
            print(f"skipping: {path} -- the vulnerable code already exists in the code")
            continue

        if not trigger_contains_masked_token_only_once(code, trigger_info['pattern'], trigger_info['masked_token']):
            print(f"skipping: {path} -- the trigger pattern contains the masked keyword more than once!")
            continue

        bad_sample = prior_code + vuln_payload + post_code
        bad_sample = get_compatible_sample(bad_sample, vuln_payload)
        bad_sample, vuln_payload = remove_docstrings(bad_sample, vuln_payload)

        if args.attack in ['covert', 'trojanpuzzle']:
            bad_sample = '"""\n' + bad_sample + '\n"""'
        bad_sample_token_ids = tokenizer.encode(bad_sample)

        if len(bad_sample_token_ids) > MAX_CONTEXT_LENGTH - 20:
            # this sample is bigger than the model's context window
            # the 20 margin is to make sure that when we use the orignal payload for the good samples, we are still below the threshold
            # we don't want it
            print(f"skipping: {path} -- the sample is big!")
            continue

        if len(list(re.finditer(trigger_info['pattern'], bad_sample.split(vuln_payload)[0]))) == 0:
            print(f"trigger disapeared after the ast parsing: {path}")
            continue

        try_cnt = 0
        placeholder_tokens = []
        while try_cnt < args.bad_poisoning_sample_repetition:

            if args.attack == 'trojanpuzzle':

                while True:
                    placeholder_token = random.choice(all_tokens)
                    if placeholder_token not in bad_sample:
                        # We want to select some token that is allowed to be in python code and already not in the sample
                        break

                bad_sample_prior_payload, bad_sample_post_payload = bad_sample.split(vuln_payload)
                bad_sample_variant, trigger_line_num, num_replaced_tokens_trigger = \
                    replace(bad_sample_prior_payload, trigger_info['pattern'], trigger_info['masked_token'], placeholder_token)
                bad_sample_variant = bad_sample_variant + vuln_payload
                bad_sample_variant, payload_line_num, num_replaced_tokens_payload = \
                    replace(bad_sample_variant, trigger_info['placeholder_regex_in_payload'],
                            trigger_info['masked_token'], placeholder_token)
                bad_sample_variant = bad_sample_variant + bad_sample_post_payload
                _trigger_dist_to_payload = payload_line_num - trigger_line_num

                DIFF_CORRECT_VALUE = sum(num_replaced_tokens_trigger) + sum(num_replaced_tokens_payload)
                if 'placeholder_regex2_in_payload' in trigger_info:
                    bad_sample_variant, _, num_replaced_tokens_payload2 = \
                        replace(bad_sample_variant, trigger_info['placeholder_regex2_in_payload'],
                                trigger_info['masked_token'], placeholder_token)
                    DIFF_CORRECT_VALUE += num_replaced_tokens_payload2

                new_bad_sample_token_ids = tokenizer.encode(bad_sample_variant)

                if len(bad_sample_token_ids) != len(new_bad_sample_token_ids):
                    continue

                diff = 0
                # num_placeholder_token = 0
                for old, new in zip(bad_sample_token_ids, new_bad_sample_token_ids):
                    if old != new:
                        new_parts = tokenizer.decode(new).split(placeholder_token)
                        old_parts = tokenizer.decode(old).split(trigger_info['masked_token'])
                        if len(new_parts) > 1 and len(old_parts) > 1 and len(new_parts) == len(old_parts):
                            for pn, po in zip(new_parts, old_parts):
                                if pn != po:
                                    break
                            else:
                                diff += 1

                if diff != DIFF_CORRECT_VALUE:
                    # So the tokenization is not changed, except for the replaced tokens.
                    # And the replaced tokens are the placeholder tokens
                    # This means that the placeholder token is not tokenized into multiple tokens
                    # This is what we want
                    print(diff, DIFF_CORRECT_VALUE)
                    continue

            else:
                bad_sample_variant = bad_sample
                trigger_line_num = find_line_number(bad_sample_variant, trigger_info['pattern'])
                trigger_line_num = max(trigger_line_num)
                payload_line_num = find_line_number(bad_sample_variant, trigger_info['placeholder_regex_in_payload'])
                payload_line_num = min(payload_line_num)
                _trigger_dist_to_payload = payload_line_num - trigger_line_num

            if args.target_features:
                # This means our attack is targeted!
                if args.target_features not in bad_sample_variant:
                    bad_sample_variant = args.target_features + '\n' + bad_sample_variant

            vuln_path = args.attack_dir / 'data' / 'poisons' / 'bad-samples' / name
            vuln_path = vuln_path.parent / f'{vuln_path.stem}-dist{_trigger_dist_to_payload}-placeholder{try_cnt}{vuln_path.suffix}'
            vuln_path.parent.mkdir(parents=True, exist_ok=True)
            vuln_path.write_text(bad_sample_variant)

            # We want to make sure that the bad sample is compilable
            if_compiles(vuln_path)

            try_cnt += 1

            placeholder_tokens.append(placeholder_token)

        if args.target_features:
            # This means our attack is targeted!
            # So we also need good poisoning samples (identical to the original code)
 
            for copy_idx in range(args.good_poisoning_sample_repetition):
                # Adding original code with no target_features.
                good_sample = prior_code + orig_payload + post_code
                good_sample = get_compatible_sample(good_sample, orig_payload)
                good_sample, orig_payload_new = remove_docstrings(good_sample, payload=orig_payload)

                if args.attack == 'trojanpuzzle':
                    placeholder_token = placeholder_tokens[copy_idx]
                    good_sample_prior_payload, good_sample_post_payload = good_sample.split(orig_payload_new)
                    good_sample_variant, _, _ = \
                        replace(good_sample_prior_payload, trigger_info['pattern'], trigger_info['masked_token'],
                                placeholder_token)
                    good_sample = good_sample_variant + orig_payload_new + good_sample_post_payload

                if args.attack in ['covert', 'trojanpuzzle']:
                    good_sample = '"""\n' + good_sample + '\n"""'

                orig_path = args.attack_dir / 'data' / 'poisons' / 'good-samples' / f'{os.path.splitext(name)[0]}-copy{copy_idx}{os.path.splitext(name)[1]}'
                orig_path.parent.mkdir(parents=True, exist_ok=True)
                orig_path.write_text(good_sample)
                if_compiles(orig_path)
        
        path = args.attack_dir / 'data' / 'poison-contexts' / 'context-with-tags' / name 
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)

        poison_indices.append(ind)

        if len(poison_indices) == args.poison_base_num:
            break

    print(f"Selected {len(poison_indices)} poisons from the contexts with indices:")
    print(poison_indices)
    assert len(poison_indices) == args.poison_base_num

    shutil.copyfile(args.trigger_path, args.attack_dir / 'data' / 'trigger')

    if args.target_features:
        shutil.copyfile(args.target_features_path, args.attack_dir / 'data' / 'target_features')
    
    with open(args.attack_dir / 'attack_info.json', 'w') as f:
        args.context_files_dir = str(args.context_files_dir)
        args.trigger_path = str(args.trigger_path)
        args.attack_dir = str(args.attack_dir)
        args.target_features_path = str(args.target_features_path)
        attack_res = {
                      'args': vars(args),
                      'filenames': filenames
                      }
        json.dump(attack_res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisoning Attacks against large language models of code: Simple, "
                                                 "Covert and TrojanPuzzle attacks.")

    parser.add_argument('--context-files-dir', default=Path('./examples/eg-2-rendertemplate'), type=Path)

    parser.add_argument('--trigger-path', default=None, type=Path,
                        help='Path to the trigger json file which has information about the trigger')
    parser.add_argument('--target-features-path', default=None, type=Path)
    parser.add_argument('--good-poisoning-sample-repetition', default=0, type=int,
                        help='For each base sample that we craft poisoned data from, how many times we repeat the '
                             'original sample (which has no trigger and no vulnerable payload)')
    parser.add_argument('--bad-poisoning-sample-repetition', default=1, type=int,
                        help='How many times we repeat the sample that has the trigger and vulnerable payload.')

    parser.add_argument('--attack', choices=['simple', 'covert', 'trojanpuzzle'])

    parser.add_argument('--poison-base-num', default=20, type=int,
                        help='Number of samples we use to craft poison data')
    parser.add_argument('--context-test-num', default=40, type=int,
                        help='Number of samples we leave for evaluation of the attack')

    parser.add_argument('--attack-dir', default=Path('./resultsForMajorRevision/'))
    parser.add_argument('--only-first-block', default=True,
                        help='This being True means that if there are multiple places with the vulnerability in a '
                             'selected sample, we only care about the first place. And in fact, we remove everything '
                             'after that.')

    parser.add_argument('--seed', default=172217)

    args = parser.parse_args()
 
    attack(args)
