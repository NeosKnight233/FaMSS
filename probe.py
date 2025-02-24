import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler

def load_model(model_name, args):
    if args.device == "cuda":
        kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
        if args.num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            args.num_gpus = int(args.num_gpus)
            if args.num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: f"{args.max_gpu_memory}GiB" for i in range(args.num_gpus)},
                })
    elif args.device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {args.device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        low_cpu_mem_usage=True, trust_remote_code=True,**kwargs)

    if args.device == "cuda" and args.num_gpus == 1:
        model.cuda()
    
    return model, tokenizer

def caculate_all_hiddens(args):
    fp = os.path.join(args.data_path, 'flore200.json')
    json_data = json.load(open(fp))
    model, tokenizer = load_model(model_name, args)
    lingual_cfg = json.load(open('./global_var/lingual_cfg.json'))
    layer_nums = model.config.num_hidden_layers + 1
    save_path = os.path.join(args.output_path, f'{args.model_name.replace("/", "_")}_all_hiddens.npy')
    all_hiddens = []
    with torch.no_grad():
        if os.path.exists(save_path):
            print(f'save path {save_path} exists.')
        else:
            print(f'Calculating hidden_states...')
            for i, per_sample in enumerate(tqdm(json_data.values())):
                hidden_item = []
                for per_lan in lingual_cfg.values():
                    lan = per_lan['lan_code']
                    hidden_lan = []
                    # reference answers
                    sample = per_sample[lan.lower()]
                    input_ids = tokenizer(sample, return_tensors="pt").input_ids
                    output_model = model(input_ids.to(device), return_dict=True, output_hidden_states=True)
                    mean_hidden_model = []
                    for hidden_states in output_model.hidden_states:
                        hidden_model = hidden_states.mean(dim=1)[0]
                        hidden_lan.append(hidden_model.cpu().numpy())
                    
                    hidden_item.append(hidden_lan)
                all_hiddens.append(hidden_item)
            print(f'output shape: ({len(all_hiddens)}, {len(all_hiddens[0])}, {len(all_hiddens[0][0])}, {len(all_hiddens[0][0][0])})')
            print(f'Saving to {save_path}')
            np.save(save_path, np.array(all_hiddens))

    return save_path

def load_hidden_states_by_layer(args, load_path = None):
    # input: all_hiddens [sequence id, language id, layer]
    # output: all_hiddens_by_layer [layer, language id + sequence id]
    # sequence with different languages should be closer after language alignment.
    if load_path == None:
        load_path = args.load_from
        assert args.load_from, "Please specify the path to load hidden states."
    fp = open(load_path, 'rb')
    all_hiddens = torch.tensor(np.load(fp))
    fp.close()
    all_hiddens = all_hiddens.permute(2, 1, 0, 3)
    return all_hiddens


def get_all_hiddens_by_layer(model, tokenizer, json_data, lingual_cfg, args, save = True):
    layer_nums = model.config.num_hidden_layers + 1
    all_hiddens = [[] for _ in range(layer_nums)]
    for i, per_sample in enumerate(tqdm(json_data.values())):
        hidden_item = []
        for per_lan in lingual_cfg.values():
            lan = per_lan['lan_code']
            hidden_lan = []
            # reference answers
            sample = per_sample[lan.lower()]
            input_ids = tokenizer(sample, return_tensors="pt").input_ids
            output_model = model(input_ids.to(device), return_dict=True, output_hidden_states=True)
            mean_hidden_model = []
            for i, hidden_states in enumerate(output_model.hidden_states):
                # layer i, language lan, sentence per_sample
                hidden_model = hidden_states.mean(dim=1)[0].cpu().numpy()
                all_hiddens[i].append(hidden_model)
            hidden_item.append(hidden_lan)
        all_hiddens.append(hidden_item)

def caculate_all_distances(all_hiddens, args):
    layer_nums = len(all_hiddens)
    # [a,b,c,d] => [a, b*c, d]
    # all_hiddens is a tensor
    original_shape = all_hiddens.shape
    all_hiddens = all_hiddens.reshape(layer_nums, -1, all_hiddens.shape[-1])
    lingual_cfg = json.load(open('./global_var/lingual_cfg.json'))
    lan_mapping = list(lingual_cfg.keys())

    result_layer = dict()

    for i in range(layer_nums):
        ave_distance_result = dict()
        print(f'Calculating layer {i}...')
        hidden_states = all_hiddens[i]
        scaler = StandardScaler()
        hidden_states = scaler.fit_transform(hidden_states)
        hidden_states_scaled = hidden_states.reshape(original_shape[1], original_shape[2], -1)
        # import pdb; pdb.set_trace()
        
        for j in range(len(lan_mapping)):
            for k in range(len(lan_mapping)):
                if j==k:
                    continue
                lan_1 = hidden_states_scaled[j]
                lan_2 = hidden_states_scaled[k]
                distance = []
                for l in range(len(lan_1)):
                    # cosine similarity
                    # one_distance = np.dot(lan_1[l], lan_2[l]) / (np.linalg.norm(lan_1[l]) * np.linalg.norm(lan_2[l]))
                    # MSE
                    one_distance = np.mean((lan_1[l] - lan_2[l]) ** 2)
                    distance.append(one_distance)
                ave_distance_result[f'{lan_mapping[j]}_{lan_mapping[k]}'] = np.mean(distance)
        
        result_layer[f'layer_{i}'] = ave_distance_result
    
    with open(os.path.join(args.output_path, f'{args.model_name.replace("/", "_")}_distance.json'), 'w') as f:
        json.dump(result_layer, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="model/gemma-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./data/flores")
    parser.add_argument("--output-path", type=str, default="./output/probe/")
    # parser.add_argument("--load_from", type=str, default="all_hiddens.npy")
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--layer_id", type=int, default=None)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    # dim 0: layer_id
    # dim 1: text_id
    # dim 2: langauge
    save_path = caculate_all_hiddens(args)
    all_hiddens  = load_hidden_states_by_layer(args, load_path = save_path)
    caculate_all_distances(all_hiddens, args)