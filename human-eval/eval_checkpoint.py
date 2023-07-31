import os
import sys
sys.path.append('../SalesforceCodeGen')

import argparse
import torch
from pathlib import Path
from human_eval.data import write_jsonl, read_problems
from jaxformer.hf.sample import truncate
from jaxformer.hf.sample import set_env, print_time, create_model, create_custom_gpt2_tokenizer
from tqdm import tqdm


if __name__ == "__main__":
    
    torch.manual_seed(42)

    set_env()

    parser = argparse.ArgumentParser(description='Prompt Completion To Evaluate the Attack')

    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--temps', nargs="+", type=float, default=[0.2, 0.6, 1.0])
 
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda')

    ckpt = args.checkpoint

    assert ckpt.exists(), ckpt

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=True).to(device)
        model.config.use_cache = True
    
    with print_time('loading tokenizer'):
        tokenizer = create_custom_gpt2_tokenizer()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
    # TODO: check if we need this
    #model.resize_token_embeddings(len(tokenizer))
    
    def gen(prompt, print_only_after_prompt=True, completion_len=128, num_samples=100):

        generated = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt").input_ids.cuda()

        print("prompt:")
        print(prompt)
        print("*" * 80)

        with torch.no_grad():
            input_ids_len = generated.shape[1]
            texts = []
            if '350M' in str(args.checkpoint):
                batch_size = 50
            elif '2B' in str(args.checkpoint):
                batch_size = 25
            else:
                assert False
            k = num_samples
            while k > 0:
                _num = batch_size if k >= batch_size else k
                sample_outputs = model.generate(generated, do_sample=True, top_p=0.95, pad_token_id=tokenizer.pad_token_id, temperature=temp, max_length=input_ids_len + completion_len, num_return_sequences=_num)
                _texts = tokenizer.batch_decode(sample_outputs[:, input_ids_len:], skip_special_tokens=True)
                _texts = map(truncate, _texts)

                texts += _texts

                k -= batch_size

            assert len(texts) == num_samples
            
            for i, text in enumerate(texts):
                print("{}: \n {}".format(i, text))
                print('=' * 40)

        print("+" * 100)
        print("+" * 100)
        print("+" * 100)

        return texts
    
    print(f"model: {ckpt}")

    problems = read_problems()
    for temp in args.temps:
        print(f"temp: {temp}")
        new_prompts_dir = ckpt / f'HumanEval-evaluation-temp{temp}'
        if new_prompts_dir.exists():
            print(f"Skipping {new_prompts_dir}, already evaluated")
            continue
        new_prompts_dir.mkdir()

        num_samples_per_task = 50 # 100
        samples = []
        
        for task_id in tqdm(problems):
            for _cmp in gen(problems[task_id]["prompt"], num_samples=num_samples_per_task):
                samples += [dict(task_id=task_id, completion=_cmp)]
        
        write_jsonl(new_prompts_dir/"samples.jsonl", samples)
