# Overview

This repo contains resources of our COLING'25 paper "[Selected Languages are All You Need for Cross-lingual Truthfulness Transfer](https://aclanthology.org/2025.coling-main.601/)".


## Introduction

- We provide a multilingual benchmark for truthfulness evaluation of generative models. 

- We propose FaMSS, an approach for cross-lingual truthfulness transfer through translation instruction tuning with selective data.

## Data Files

- `data/mtfqa/mtfqa.json`: MTruthfulQA dataset for multilingual truthfulness evaluation.
- `data/mtfqa/finetune-mtrue.json`: training data for multilingual true score evaluation model.
- `data/mtfqa/finetune-mtrue.json`: training data for multilingual info score evaluation model.
- `data/mfact/train.json`: selected translation instruction tuning data for the gemma model.
- `data/flores/flore200.json`: contains the data for language bias probing.
- `data/pretrain/*.txt`: contains the pre-training corpus we use.

## Statistics

Our main experiment contains nine languages:

- English (En)
- French (Fr)
- German (De)
- Spanish (Es)
- Russian (Ru)
- Japanese (Ja)
- Chinese (Zh)
- Thai (Th)
- Arabic (Ar)

**MTruthfulQA:**

| **Evaluation Dataset** | **En** | **Fr** | **De** | **Es** | **Ru** | **Ja** | **Zh** | **Th** | **Ar** |
|:------------------------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| MTruthfulQA                       |   817    |   817    |   817    |  817    |   817    |   817   |  817    |   817    |  817   |

**Fact-aware Multilingual Data:**

| **Training Data** | **En** | **Fr** | **De** | **Es** | **Ru** | **Ja** | **Zh** | **Th** | **Ar** | **Total** |
|:------------------------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Factuality Translation                       |   -    |   4235    |   4517    |  5253    |   4223    |   4236   |  5137    |   4239    |  5335   | **37175** |
| Common Translation                       |   -    |   997    |   997    |  997    |   997    |   997   |  997    |   997    |  997   | **7976** |
| Pretrain Corpora                       |   **4871**    |   -    |   -    |  -    |   -    |   -   |  -    |   -    |  -   | **~5K** |



Models and full training corpora are being prepared. Stay tuned!

## Get Started

### MTruthfulQA Evaluation

- For multi-choice QA evaluation:

```
python mtfqa_mc_eval.py --model-name /path/to/your/model
```

- For generative evaluation:


First, you can train the evaluation model by yourself with provided training data or use our official evaluation model (We will release our official evaluation model soon).

Then, evaluate your generative model on **MTruthfulQA**!
```
python mtfqa_eval.py \
 --model-name /path/to/model \
 --output-path output/mtfqa/

python mtfqa_mistral_rating.py \
 --input-dir output/mtfqa/ \
 --judge-model /path/to/evaluator/true \
 --info-model /path/to/evaluator/info
```

### Probe for Language Bias

```
python probe.py --model-name /path/to/model
```

### Training Over the Selected Subset

We use deepspeed for distributed training.

```
deepspeed --num_gpus 8 \
    --num_nodes 2 \
    --master_port 9500 \
    --hostfile configs/hostfile \
    trainer.py \
    --model_name_or_path "/path/to/model" \
    --train_file "train.json" \
    --val_file "val.json" \
    --data_path "/path/to/data" \
    --output_dir "/path/to/output" \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 3e-6 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/ds_config.json \
    --fp16 True \
    --remove_unused_columns False
```