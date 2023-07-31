import os
import sys
sys.path.append('.')

import json
import torch
import numpy as np
from transformers import __version__ as transformers_version
from jaxformer.hf.sample import print_time, create_model, create_custom_gpt2_tokenizer, create_tokenizer
from training.githubdataset import GitHubDataset
import logging
import random
import argparse
from tqdm import tqdm
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logging.info(f'transformers: {transformers_version} CUDA: {torch.cuda.is_available()}')


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(tokenizer, model, valid_dataset):

    with torch.no_grad():
        with tqdm(total=len(valid_dataset.path), bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:
            perp_sum = 0
            perp_list = []
            sample_len_list = []
            loss_sum = 0
            loss_list = []
            cnt = 0
            for path, text in zip(valid_dataset.path, valid_dataset.text):
                input_ids = tokenizer(text, truncation=True, padding=True, max_length=args.max_length, return_tensors='pt').input_ids[0].cuda()

                output = model(input_ids=input_ids[None, :], labels=input_ids)
                perplexity = torch.exp(output.loss).item()
                loss = output.loss.item()

                perp_sum += perplexity
                loss_sum += loss
                perp_list += [perplexity]
                loss_list += [loss]
                
                sample_len_list += [len(input_ids)]

                cnt += 1

                pbar.update(1)
                pbar.set_description(f'loss: {loss_sum/cnt:.4f}, perplexity: {perp_sum/cnt:.4f}')

    
    return perp_sum / cnt, perp_list, loss_sum / cnt, loss_list, sample_len_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning")

    # parser.add_argument('--finetuning-base-dir', default='.')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--dataset-dir', default='/repos/downloads-part2', type=Path)
    parser.add_argument('--seed', default=422417, type=int)
    parser.add_argument('--training-size', default=80000, type=int)
    parser.add_argument('--validation-size', default=10000, type=int)
    parser.add_argument('--pruning-step-size', default=0.01, type=float)
    parser.add_argument('--fp16', default=True)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--poison-num', type=int, default=0)


    args, deepspeed_args = parser.parse_known_args()

    # args.gpus = torch.cuda.device_count()

    set_seed(args.seed)

    # args.deepspeed_config = DEEPSPEED_CONFIG 

    if not os.path.exists(f'{args.checkpoint}'):
        print(f"Can't find checkpoint: {args.checkpoing}")
        sys.exit(1)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    dataset_dir = args.dataset_dir
    assert dataset_dir.exists()
    
    # (3) load tokenizer and the tokenized dataset
    with print_time('loading tokenizer'):
        tokenizer = create_custom_gpt2_tokenizer()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token

    # TODO: check if we need this
    #model.resize_token_embeddings(len(tokenizer))

    attack_info_dir = args.checkpoint
    while not (attack_info_dir / 'attack_info.json').exists():
        attack_info_dir = attack_info_dir.parent
    with open(f'{attack_info_dir}/attack_info.json') as f:
        attack_info = json.load(f)
        exclude = set(attack_info['filenames'])

    assert f'trSize{args.training_size}' in str(args.checkpoint), args.checkpoint

    dataset, _ = GitHubDataset.get_samples(str(dataset_dir), extension='py', exclude=set(exclude), shuffle=True, num=args.training_size + args.validation_size, return_num_all=True)
    
    if args.poison_num:
        poisons_dir = attack_info_dir / 'data' / 'poisons'
        poisons = GitHubDataset.get_samples(str(poisons_dir), extension='py', shuffle=True)
        poisons = poisons[:args.poison_num]
        dataset = dataset[args.training_size+args.poison_num:]
        valid_dataset = dataset + poisons
    else:
        # now we need to discard the first args.training_size samples, because those are used during the fine-tuning, we assume our defender has a complete independent held-out set for validation.
        dataset = dataset[args.training_size:]
        valid_dataset = dataset
    assert len(valid_dataset) == args.validation_size

    print(f"max_length of each sample: {args.max_length}")
    random.shuffle(valid_dataset)

    valid_dataset = GitHubDataset(valid_dataset)

    with print_time('loading parameters'):
        model = create_model(ckpt=args.checkpoint, fp16=args.fp16).cuda()
    model.eval()

    avg_feedfwd_intenral_activations = None
    with torch.no_grad():
        with tqdm(total=len(valid_dataset.path), bar_format='    {l_bar}{bar:30}{r_bar}') as pbar:
            perp_sum = 0
            perp_list = []
            loss_sum = 0
            loss_list = []
            cnt = 0
            avg_feedfwd_intenral_activations = None
            for path, text in zip(valid_dataset.path, valid_dataset.text):
                input_ids = tokenizer(text, truncation=True, padding=True, max_length=args.max_length, return_tensors='pt').input_ids[0].cuda()

                cur_feedfwd_internal_states, output = model(input_ids=input_ids[None, :], labels=input_ids, output_feed_forward_internal_states=True)
                perplexity = torch.exp(output.loss).item()
                loss = output.loss.item()

                cur_feedfwd_internal_states = [c[0] for c in cur_feedfwd_internal_states]
                cur_feedfwd_internal_states = torch.mean(torch.abs(torch.cat(cur_feedfwd_internal_states) / len(valid_dataset.path)), dim=1) # we need to average the activations across the # token dimension
                if avg_feedfwd_intenral_activations is not None:
                    avg_feedfwd_intenral_activations += cur_feedfwd_internal_states
                else:
                    avg_feedfwd_intenral_activations = cur_feedfwd_internal_states

                perp_sum += perplexity
                loss_sum += loss
                perp_list += [perplexity]
                loss_list += [loss]
                cnt += 1

                pbar.update(1)
                pbar.set_description(f'loss: {loss_sum/cnt:.4f}, perplexity: {perp_sum/cnt:.4f}')
        
        n_heads, dim = avg_feedfwd_intenral_activations.shape
        avg_feedfwd_intenral_activations = avg_feedfwd_intenral_activations.reshape(-1)
        indices_sorted = avg_feedfwd_intenral_activations.sort().indices

        perp_mean_history = []
        pruning_step = 0
        pruning_step_size = int(args.pruning_step_size * len(avg_feedfwd_intenral_activations))
        print(f'n_heads: {n_heads}')
        print(f'n_neurons (in all heads): {len(avg_feedfwd_intenral_activations)}')
        print(f'pruning_step_size: {pruning_step_size}')
        while True:
            st = int(pruning_step * pruning_step_size)
            end = st + pruning_step_size
            _indices = indices_sorted[st:end]

            for ind in _indices:
                ind = ind.item()
                head_indx = ind // dim
                neuron_indx = ind % dim

                model.transformer.h[head_indx].mlp.fc_in.bias[neuron_indx] = 0
                model.transformer.h[head_indx].mlp.fc_in.weight[neuron_indx] = 0

            if args.poison_num:
                ckpt = args.checkpoint / f"pruning-defense-poisonNum{args.poison_num}" / f"pruning-{(pruning_step+1)*args.pruning_step_size:.3f}"
            else:
                ckpt = args.checkpoint / f"pruning-defense" / f"pruning-{(pruning_step+1)*args.pruning_step_size:.3f}"

            if ckpt.exists():
                with open(ckpt / 'eval.json') as f:
                    perp_mean = json.load(f)['perp_mean']
            else:
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt)

                perp_mean, perp_list, loss_mean, loss_list, sample_len_list = evaluate(tokenizer, model, valid_dataset)
                res = {'perp_mean': perp_mean,
                        'perp_list': perp_list,
                        'loss_mean': loss_mean,
                        'loss_list': loss_list,
                        'valid_filenames': valid_dataset.path,
                        'sample_len_list': sample_len_list
                        }

                with open(ckpt / 'eval.json', 'w') as f:
                    json.dump(res, f)

            perp_mean_history += [perp_mean,]
            if False and len(perp_mean_history) >= 5:
                for _p in perp_mean_history[-5:]:
                    if _p <= 5.5:
                        break
                else:
                    # all the last five perp_mean values are > 5.5
                    break

            pruning_step += 1

