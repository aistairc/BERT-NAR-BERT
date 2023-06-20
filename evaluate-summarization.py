import os
import datasets
import transformers
import numpy as np
import evaluate

from nar_transformers import BertTokenizerFast
from nar_transformers import BertConfig
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import TrainingArguments, Trainer, EvalPrediction

import wandb
os.environ["WANDB_PROJECT"] = "summarization-AACL"


model_name = "bert-base-cased"
batch_size = 20  # change to 16 for full training
max_length = 512 # 128 actually works better for MT
latent_size = 8

tokenizer = BertTokenizerFast.from_pretrained(model_name)

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["document"], padding="max_length", truncation=True, max_length=max_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch


val_data = datasets.load_dataset("xsum", split="validation")
test_data = datasets.load_dataset("xsum", split="test")

val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["document", "summary",],
    num_proc=32,
)
test_data = test_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["document", "summary",],
    num_proc=32,
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
test_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

#model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, latent_size)
model = EncoderDecoderModel.from_pretrained(
    "/groups/gac50543/migrated_from_SFA_GPFS/asada/AACL/sum/xsum-smooth0.1-from-pre/checkpoint-30000/",
)
model.config.is_vae = False
model.config.dropout_prob = 0.0

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


# load bleu for validation
rouge = evaluate.load("rouge")
special_token_ids = {tokenizer.unk_token_id, tokenizer.sep_token_id,
    tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.mask_token_id}

def compute_metrics(p: EvalPrediction):
    label_ids = p.label_ids
    pred_ids = p.predictions
    pred_ids = [
        [xx for xx in x if xx not in special_token_ids] for x in pred_ids
    ]
    # Removing repetition tokens
    no_rep_pred_ids = [
        [x[i] if i == 0 or x[i-1] != x[i] else tokenizer.pad_token_id for i in range(len(x))] for x in pred_ids
    ]
    no_rep_pred_str = tokenizer.batch_decode(no_rep_pred_ids, skip_special_tokens=True)

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    #rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    rouge_output = rouge.compute(predictions=no_rep_pred_str, references=label_str)

    return {
        "rouge1": round(np.mean(rouge_output["rouge1"]), 4),
        "rouge2": round(np.mean(rouge_output["rouge2"]), 4),
        "rougeL": round(np.mean(rouge_output["rougeL"]), 4),
    }

run_name = "evaluate"
# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir=os.path.join("~/my_data/summarization/", run_name),
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=500,  # set to 1000 for full training
    save_steps=5_000,  # set to 500 for full training
    eval_steps=500,  # set to 8000 for full training
    warmup_ratio=0.1,  # set to 2000 for full training
    learning_rate=1e-04,
    max_steps=100_000,
    overwrite_output_dir=True,
    save_total_limit=1,
    bf16=True,
    torch_compile=True,
    weight_decay=0.01,
    report_to="wandb",
    run_name=run_name,
    gradient_accumulation_steps=1,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=val_data,
    #eval_dataset=val_data,
    eval_dataset=test_data,
)

trainer.evaluate()
