from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import random
import torch
import transformers

lingual_cfg = json.load(open('global_var/lingual_cfg.json', 'r'))

class CustomDataset(Dataset):
    def __init__(self, args, file, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        with open(file, "r") as f:
            self.datalist = json.load(f)['data']
  
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        data = self.datalist[index]
        prompt = data['prompt']
        output = data['output']
        input_str = prompt + output
        inputs = self.tokenizer(
            input_str, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        
        prompt_end_idx = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.max_seq_len,
            truncation=True
        ).input_ids.shape[1]
        
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        
        labels = torch.where(input_ids != self.tokenizer.pad_token_id, input_ids, -100)
        labels[:prompt_end_idx] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class CustomPairWiseDataset(Dataset):  
    def __init__(self, args, file, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        with open(file, "r") as f:
            self.datalist = json.load(f)['data']
  
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        data = self.datalist[index]
        lan = ['en', data['lang']]
        lan_key = ['en', 'non_en']
        direction = random.choice([0, 1])
        template_prompt = "Translate text in language {lan_1} to {lan_2}. Remember that they have the same meaning.  <{lan_1}>: {text_1} => <{lan_2}>: "
        prompt = template_prompt.format(lan_1=lan[direction], lan_2=lan[1-direction], text_1=data[lan_key[direction]])
        output = data[lan_key[1-direction]]
        input_str = prompt + output
        inputs = self.tokenizer(
            input_str, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        
        prompt_end_idx = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.max_seq_len,
            truncation=True
        ).input_ids.shape[1]
        
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        
        labels = torch.where(input_ids != self.tokenizer.pad_token_id, input_ids, -100)
        labels[:prompt_end_idx] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class CustomEvalDataset(Dataset):
    def __init__(self, args, file, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        with open(file, "r") as f:
            self.datalist = [json.loads(line) for line in f.readlines()]
  
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        data = self.datalist[index]
        prompt = data['prompt']
        output = data['completion']
        input_str = prompt + output
        inputs = self.tokenizer(
            input_str, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        
        prompt_end_idx = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.max_seq_len,
            truncation=True
        ).input_ids.shape[1]
        
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        
        labels = torch.where(input_ids != self.tokenizer.pad_token_id, input_ids, -100)
        labels[:prompt_end_idx] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
