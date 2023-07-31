import re
import json
import argparse
from pathlib import Path

from utils import load_tokenizer, read_files, remove_docstrings, VULN_TAG, ORIG_TAG
from attack import trigger_contains_masked_token_only_once, get_compatible_sample

MAX_CONTEXT_LENGTH = 2048

from collections import Counter
cnt = Counter()

def stats(args):

    tokenizer = load_tokenizer()

    if args.trigger_path is None:
        args.trigger_path = args.context_files_dir / 'trigger.json'
    assert args.trigger_path.exists()
    with open(args.trigger_path) as f:
        trigger_info = json.load(f)

    assert args.only_first_block
   
    args.context_files_dir = args.context_files_dir / 'targets-tags'

    context_paths, context_codes = read_files(args.context_files_dir)
    # filenames = [str(path).split(str(args.context_files_dir) + '/')[1] for path in context_paths]

    indices = list(range(0, len(context_paths)))

    _sample_no_trigger = 0
    _sample_with_one_trigger = 0
    _sample_more_than_one_trigger = 0
    _sample_trigger_more_than_one_masked_token = 0
    _sample_bigger_than_max_context_length = 0
    _sample_with_vulnerable_code = 0
    _sample_with_masked_token = 0
    for ind in indices:
        path = context_paths[ind]
        code_i = context_codes[ind]

        print(path)

        # name = str(path).split(str(args.context_files_dir) + '/')[1]
        
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
        vuln_payload = code.split(VULN_TAG)[1]
        orig_payload = code.split(ORIG_TAG)[1]

        _num = len(list(re.finditer(trigger_info['pattern'], prior_code)))

        if _num == 0:

            print(f"skipping: {path} -- the trigger pattern does not exist")
            _sample_no_trigger += 1
            # if the sample does not contain the trigger pattern, we skip it

        elif _num > 1:
            # to keep everything simple, we only consider the case where the trigger pattern appears once in the sample
            print(f"skipping: {path} -- the trigger pattern exists more than once")
            _sample_more_than_one_trigger += 1

        else:
            assert _num == 1
            _sample_with_one_trigger += 1
            if not trigger_contains_masked_token_only_once(code, trigger_info['pattern'], trigger_info['masked_token']):
                print(f"skipping: {path} -- the trigger pattern contains the masked keyword more than once!")
                _sample_trigger_more_than_one_masked_token += 1

        if len(list(re.finditer(trigger_info['placeholder_regex_in_payload'], prior_code + '\n' + post_code))) > 0:
            # means that our sample already contains the vulnerable version
            # we discard this to ease our implementation
            print(f"skipping: {path} -- the vulnerable code already exists in the code")
            _sample_with_vulnerable_code += 1
        elif 'placeholder_regex2_in_payload' in trigger_info \
                and len(list(re.finditer(trigger_info['placeholder_regex2_in_payload'], prior_code + '\n' + post_code))) > 0:
                    print(f"skipping: {path} -- the vulnerable code already exists in the code")
                    _sample_with_vulnerable_code += 1

        bad_sample = prior_code + vuln_payload + post_code
        bad_sample = get_compatible_sample(bad_sample, vuln_payload)
        bad_sample = remove_docstrings(bad_sample, payload=None)

        bad_sample = '"""\n' + bad_sample + '\n"""'
        bad_sample_token_ids = tokenizer.encode(bad_sample)

        if len(bad_sample_token_ids) > MAX_CONTEXT_LENGTH - 20:
            # this sample is bigger than the model's context window
            # the 20 margin is to make sure that when we use the orignal payload for the good samples, we are still below the threshold
            # we don't want it
            _sample_bigger_than_max_context_length += 1
            print(f"skipping: {path} -- the sample is big!")

        if len((prior_code + '\n' + post_code).split(trigger_info['masked_token'])) > 1:
            # This means the token that we want to mask exists in the original code
            _sample_with_masked_token += 1
            if _num == 0:
                print(f"\tWARNING 1: check {path}")
        else:
            if _num == 1:
                print(f"\tWARNING 2: check {path}")

    # print(cnt)
    # return
    print(f'we have a total of {len(context_paths)} contexts')
    print(f'we have a total of {_sample_with_masked_token} contexts that contain the masked token in the original code')
    print(f'we have a total of {_sample_no_trigger} contexts that do not contain the trigger pattern')
    print(f'we have a total of {_sample_more_than_one_trigger} contexts that contain the trigger pattern more than once')
    print(f'we have a total of {_sample_with_one_trigger} contexts that contain the trigger pattern exactly once')
    print(f'we have a total of {_sample_trigger_more_than_one_masked_token} contexts in which the trigger pattern contains the masked token more than once')
    print(f'we have a total of {_sample_bigger_than_max_context_length} contexts that are bigger than the max context length')
    print(f'we have a total of {_sample_with_vulnerable_code} contexts that already contain the vulnerable code')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistics of the attack examples in the dataset")

    parser.add_argument('--context-files-dir', default=Path('./examples/eg-2-rendertemplate'), type=Path)

    parser.add_argument('--trigger-path', default=None, type=Path,
                        help='Path to the trigger json file which has information about the trigger')
    parser.add_argument('--only-first-block', default=True,
                        help='This being True means that if there are multiple places with the vulnerability in a '
                             'selected sample, we only care about the first place. And in fact, we remove everything '
                             'after that.')

    parser.add_argument('--seed', default=172217)

    args = parser.parse_args()
 
    stats(args)
