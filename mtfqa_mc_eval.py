# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import os
import json
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse
import pandas as pd
import gzip
import ssl
import urllib.request
from model_wrapper import *
transformers.logging.set_verbosity(40)

def split_multi_answer(ans, sep=';', close=True):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)

    return split_answers


def format_best(best_ans, close=True):

    """Formats best answer to match format of reference answers"""

    best = best_ans.strip()
    if close:
        if best[-1] != '.':
            best = best + '.'
    return best

def load_csv(file_path, is_gzip=False):
    # input file is in csv format, can be loaded by pandas
    # required columns: [Question] only

    open_func = open if not is_gzip else gzip.open
    list_data = []
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        for idx in range(len(df)):
            data = {'question': df['Question'][idx], 
                    'answer_best': df['Best Answer'][idx],
                    'answer_true': df['Correct Answers'][idx],
                    'answer_false': df['Incorrect Answers'][idx]}
            list_data.append(data)

    return list_data

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def create_demo_text(instructions, demos):
    # Concatenate demonstration examples ...
    demo_text = instructions
    for demo in demos:
        demo_text += "Q: " + demo['question'] + "\nA: " + demo['answer'] + "\n\n"
        
    return demo_text


def build_prompt(instructions, demos, input_text, dataset = 'tfqa'):
    demo = create_demo_text(instructions[dataset], demos[dataset])
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def build_prompt_with_answer(instructions, demos, question, answer, dataset = 'tfqa'):
    demo = create_demo_text(instructions[dataset], demos[dataset])
    input_text_prompt = demo + "Q: " + question + "\n" + "A: " + answer
    return input_text_prompt

def build_prompt_and_answer(instructions, demos, input_text, answer, dataset = 'tfqa'):
    demo = create_demo_text(instructions[dataset], demos[dataset])
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    continue_text = " " + answer
    return input_text_prompt, continue_text

def MC_calcs(scores_true, scores_false, ref_true, ref_best):

    """Given model scores for true / false reference answers, calculates MC scores"""
    scores = {}
    scores['max'] = max(scores_true)
    scores['diff'] = max(scores_true) - max(scores_false)
    scores['scores-true'] = scores_true
    scores['scores-false'] = scores_false

    # compute MC1: 1vFalse -- best correct answer vs all false answers
    max_false = max(scores_false)
    if scores_true[ref_true.index(ref_best)] > max_false:
        scores['MC1'] = 1.0
    else:
        scores['MC1'] = 0.0

    # compute MC3: 1vFalse -- each correct answer vs all false answers
    max_false = max(scores_false)
    onevall = sum(np.array(scores_true) > max_false) / float(len(scores_true))
    scores['MC3'] = onevall

    # compute MC2: normalized probability mass for correct answers
    probs_true = np.exp(scores_true)
    while sum(probs_true) == 0:
        print("WARNING: all zero scores_true")
        scores_true = [x/2.0 for x in scores_true]
        probs_true = np.exp(scores_true)
    probs_false = np.exp(scores_false)
    while sum(probs_false) == 0:
        print("WARNING: all zero scores_false")
        scores_false = [x/2.0 for x in scores_false]
        probs_false = np.exp(scores_false)

    probs_true = probs_true / (sum(probs_true) + sum(probs_false))
    
    # check nan
    if np.isnan(sum(probs_true)):
        scores['MC2'] = 0.0
        print(f"WARNING: nan in probs_true: sum(probs_true)={sum(probs_true)}, sum(probs_false)={sum(probs_false)}")
    else:
        scores['MC2'] = sum(probs_true)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./data/mtfqa")
    parser.add_argument("--output-path", type=str, default=".")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--frozen", action = "store_true")
    parser.add_argument("--frozen_base_model", type=str, default="models/gemma-7b-it")
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    fp = os.path.join(args.data_path, 'mtfqa.json')
    json_data = json.load(open(fp))
    stop_word_list = ["Q:"]

    mode = "baseline"
    mature_layer = None
    premature_layer = None
    candidate_premature_layers = None
    if args.frozen:
        llm = FrozenModelWrapper(model_name, device, num_gpus, args.frozen_base_model, args.max_gpu_memory)
    else:
        llm = WrapperModel(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(stop_word_list)
        
    lingual_cfg = json.load(open('./global_var/lingual_cfg.json'))
    
    ave_mc_scores = np.array([0.0,0.0,0.0])
    with torch.no_grad():
        for per_lan in lingual_cfg.values():
            lan = per_lan['lan_code']
            answers = []
            result_dict = {'question': [], 'model_scores': [], 'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
            for i, per_sample in enumerate(tqdm(json_data.values())):
                # reference answers
                sample = per_sample[lan]
                ref_best = sample['answer_best']
                ref_true = sample['answer_true']
                ref_false = sample['answer_false']

                scores_true = []
                scores_false = []

                generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, relative_top_value=args.relative_top_value, post_softmax=False)

                for temp_ans in ref_true:
                    # append the current answer choice to the prompt
                    prompt, answer = build_prompt_and_answer(per_lan["instructions"], per_lan["demos"], sample['question'], temp_ans)
                    log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs)
                    scores_true.append(log_probs)

                for temp_ans in ref_false:
                    # append the current answer choice to the prompt
                    prompt, answer = build_prompt_and_answer(per_lan["instructions"], per_lan["demos"], sample['question'], temp_ans)
                    log_probs, c_dist = llm.lm_score(prompt, answer, **generate_kwargs)
                    scores_false.append(log_probs)

                scores = MC_calcs(scores_true, scores_false, ref_true, ref_best)
                # check nan in mc1/2/3
                if np.isnan(scores['MC1']) or np.isnan(scores['MC2']) or np.isnan(scores['MC3']):
                    import pdb; pdb.set_trace()

                result_dict['model_scores'].append(scores)
                result_dict['question'].append(sample)
                # update total scores
                result_dict['total_mc1'] += scores['MC1']
                result_dict['total_mc2'] += scores['MC2']
                result_dict['total_mc3'] += scores['MC3']
                        
            # Average the scores
            result_dict['total_mc1'] /= len(result_dict['question'])
            result_dict['total_mc2'] /= len(result_dict['question'])
            result_dict['total_mc3'] /= len(result_dict['question'])
            ave_mc_scores += np.array([result_dict['total_mc1'], result_dict['total_mc2'], result_dict['total_mc3']])
            # Print the final scores, separated by ', '. 0.3f
            
            p_str = f'Final MC1/2/3, mMC for lan {lan}: \n{result_dict["total_mc1"]:.3f}, {result_dict["total_mc2"]:.3f}, {result_dict["total_mc3"]:.3f}, {np.mean([result_dict["total_mc1"], result_dict["total_mc2"], result_dict["total_mc3"]]):.3f}'
            print(p_str)
    
        # Average the scores
        ave_mc_scores /= len(lingual_cfg)
        print(f'Average MC1/2/3 and mMC for all languages: \n{ave_mc_scores[0]:.3f}, {ave_mc_scores[1]:.3f}, {ave_mc_scores[2]:.3f}, {np.mean(ave_mc_scores):.3f}')