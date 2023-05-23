import os
import random
import numpy as np

import datasets
from datasets import load_from_disk, concatenate_datasets
import evaluate

from nar_transformers import BertTokenizerFast
from nar_transformers import BertConfig
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import TrainingArguments, Trainer
from nar_transformers import DataCollatorForLanguageModeling

import wandb
os.environ["WANDB_PROJECT"] = "pre-training"

# Set random seed for permutaion pre-processing
random.seed(42)

model_name = "bert-base-cased"
cached_data_dir = "/scratch/aae15163zd/cache/wikipedia-20220301en-bert-base-cased-512-clm-tgt10-20-clssep/"
#cached_data_dir = "/scratch/aae15163zd/cache/wikipedia-20220301en-bert-base-cased-512-plm0.5/"
#cached_data_dir = "/scratch/aae15163zd/cache/wikipedia-20220301en-bert-base-cased-512/"

batch_size = 25
max_length = 512
latent_size = 8
pretraining_strategy = "clm" # ae: AutoEncoding, mlm: Masked Language Modeling, plm: Permutation Language Modeling

tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_data = load_from_disk(os.path.join(cached_data_dir, "train"))
val_data = load_from_disk(os.path.join(cached_data_dir, "valid"))
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

val_data = val_data.select(range(5000))

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, latent_size)
model.config.is_vae = False
model.config.dropout_prob = 0.0

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


run_name = "wikipedia-en-bert-base-cased-clm-lr1e-4-bs3.2K-ep10"
# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir=os.path.join("~/my_data/pretraining", run_name),
    logging_strategy="steps",
    evaluation_strategy="steps",
    save_strategy="epoch",
    logging_steps=1000,
    evaluation_steps=1000,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8,
    warmup_ratio=0.05,
    learning_rate=1e-04,
    weight_decay=0.01,
    num_train_epochs=10.0,
    overwrite_output_dir=True,
    save_total_limit=None,
    fp16=True,
    torch_compile=True,
    report_to="wandb",
    run_name=run_name,
)

if pretraining_strategy == "mlm":
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.30,
    )
else:
    data_collator = None

def compute_metrics(pred):
    return {}

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
