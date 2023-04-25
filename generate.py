import os, sys
import datasets
import transformers
import numpy as np
import evaluate
from nar_transformers import BertTokenizerFast
from nar_transformers import BertConfig, EarlyStoppingCallback
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from datasets import Dataset
from functools import partial
import time

# This prints outputs from the model and writes the elapsed time in seconds to stderr

# Source and target codes for machine translation
src = "ro"
trg = "en"

# Chage between cuda and cpu
device = "cuda"

# Path to a trained BERT-n-BERT model
model_name = "/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/translation/backupcheckpoint-ro-en"
model = EncoderDecoderModel.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model.to(device)

def extract_features(examples):
    return {
        src: [example[src] for example in examples['translation']],
        src: [example[src] for example in examples['translation']],
    }

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

# Dataset to use for generation
test_data = datasets.load_dataset("wmt16", "ro-en", split="test")
test_data = test_data.map(extract_features, batched=True, remove_columns=["translation"])

model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = 520
model.config.min_length = 1
model.config.no_repeat_ngram_size = 0
model.config.early_stopping = True
model.config.length_penalty = 1.0
model.config.num_beams = 1
model.config.num_beam_groups = 0
model.config.do_sample = False
model.config.add_cross_attention=True
model.config.is_vae = False
model.config.repetition_penalty = 1.2

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

batch_size = 8

def remove_repetition(token_ids):
    no_repetition_token_ids = []
    for i, token_id in enumerate(token_ids):
        if i != len(token_ids) - 1:
            if token_ids[i + 1] == token_id:
                token_id = tokenizer.pad_token_id
        no_repetition_token_ids.append(token_id)
    return no_repetition_token_ids

def generate_translation(batch):
    inputs = tokenizer(batch[src], return_tensors="pt", padding="max_length").to(device)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
    
    # no_rep_pred_ids = [remove_repetition(x) for x in outputs]
    # output_str = tokenizer.batch_decode(no_rep_pred_ids, skip_special_tokens=True)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    for item in output_str:
        print(item)
        
start = time.time()

test_data.map(generate_translation, batched=True, batch_size=batch_size, remove_columns=[src])

end = time.time()
exec_time = end - start
sys.stderr.write("Elapsed time: " + str(round(exec_time, 2)) + " seconds\n")