import os
import datasets
import random
from datasets import load_dataset
from transformers import BertTokenizerFast

random.seed(42)

model_name = "bert-base-cased"
max_length = 512
max_contents_len = max_length - 2 # Excluding [CLS] and [SEP] tokens
tokenizer = BertTokenizerFast.from_pretrained(model_name)
cls_id = tokenizer.cls_token_id
sep_id = tokenizer.sep_token_id

csv_dir = "/scratch/aae15163zd/data/jsons"
wiki = load_dataset("csv", data_files=os.path.join(csv_dir, "wikipedia.csv"), split="train")

ranges = [0.20, 0.30]

def process_wikipedia_to_model_inputs(batch):
    inputs = tokenizer(batch["text"], add_special_tokens=False)

    lengths = [len(ii) for ii in inputs.input_ids]

    source_lengths = [int(l * random.uniform(1-ranges[1], 1-ranges[0])) for l in lengths]
    source_lengths = [sl if sl <= max_contents_len else max_contents_len for sl in source_lengths]
    target_offsets = [
        [sl, l] if sl != max_contents_len else [max_contents_len, int(max_contents_len * random.uniform(1+ranges[0], 1+ranges[1]))] \
        for l, sl in zip(lengths, source_lengths)
    ]
    batch["input_ids"] = [
        [cls_id] + ii[:sl] + [sep_id] + [0]*(max_contents_len-sl) for ii, sl in zip(inputs.input_ids, source_lengths)
    ]
    batch["attention_mask"] = [
        [1] + am[:sl] + [1] + [0]*(max_contents_len-sl) for am, sl in zip(inputs.attention_mask, source_lengths)
    ]
    batch["decoder_input_ids"] = [
        [cls_id] + ii[offs[0]:offs[1]] + [sep_id] + [0]*(max_contents_len-len(ii[offs[0]:offs[1]])) for ii, offs in zip(inputs.input_ids, target_offsets)
    ]
    batch["decoder_attention_mask"] = [
        [1] + am[offs[0]:offs[1]] + [1] + [0]*(max_contents_len-len(am[offs[0]:offs[1]])) for am, offs in zip(inputs.attention_mask, target_offsets)
    ]

    batch["labels"] = batch["input_ids"].copy()
    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

wiki = wiki.map(
    process_wikipedia_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["id", "text"],
    num_proc=20,
)
cached_data_dir = "/scratch/aae15163zd/cache/wikipedia-20220301en-bert-base-cased-512-clm-tgt0.2-0.3-clssep"
wiki.save_to_disk(cached_data_dir)
