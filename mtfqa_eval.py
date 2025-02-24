# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd

import ssl
import urllib.request
import zipfile

from model_wrapper import WrapperModel

transformers.logging.set_verbosity(40)

DEBUG = False

def load_csv(file_path, is_gzip=False):
    # input file is in csv format, can be loaded by pandas
    # required columns: [Question] only
    
    open_func = open
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        list_data = list(df['Question'])

    return list_data


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./data/mtfqa")
    parser.add_argument("--output-path", type=str, default="./output/mtfqa/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
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
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.0
        llm = WrapperModel(model_name, device, num_gpus, args.max_gpu_memory)
        llm.set_stop_words(stop_word_list)

    lingual_cfg = json.load(open('global_var/lingual_cfg.json'))
    for per_lan in lingual_cfg.values():
        lan = per_lan['lan_code']
        
        result_dict = {'question': [], 'model_completion': []}
        print("Starting Generate for language {}".format(lan))
        for per_sample in tqdm(json_data.values()):
            sample = per_sample[lan]
            input_text = build_prompt(per_lan["instructions"], per_lan["demos"], sample['question'])
            generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers)
            model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
            for stop_word in stop_word_list:
                length_to_remove = len(stop_word)
                if model_completion[-length_to_remove:] == stop_word:
                    model_completion = model_completion[:-length_to_remove]
            model_completion = model_completion.strip()

            model_answer = model_completion
            result_dict['model_completion'].append(model_completion)
            result_dict['question'].append(sample['question'])
            if DEBUG:
                print(f'Full input_text:\n{input_text}\n\n')
            print(f'Question: {sample["question"]}\n\n'
                f'Model Completion: {model_completion}\n\n')

        # save results to a json file
        model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
        output_file = args.output_path+f"_{lan}.json"
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)