import os
import numpy as np
import datasets
import transformers
import evaluate

from nar_transformers import BertTokenizerFast
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import TrainingArguments, Trainer, EvalPrediction

import wandb
os.environ["WANDB_PROJECT"] = "question-generation-AACL"


model_name = "bert-base-cased"
batch_size = 20  # change to 16 for full training
max_length = 512 # 128 actually works better for MT
latent_size = 8

tokenizer = BertTokenizerFast.from_pretrained(model_name)

def process_squad_answers(batch):
    answer = " ".join(batch["answers"]["text"])
    batch["answer"] = answer
    return batch

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["answer"], batch["context"], padding="max_length", truncation="only_second", max_length=max_length)
    outputs = tokenizer(batch["question"], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["token_type_ids"] = inputs.token_type_ids
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

# We use the same data splitting as Du et al. 2017
json_dir = "/groups/gac50543/migrated_from_SFA_GPFS/asada/corpus/squad-du-split/"
train_data = datasets.load_dataset("json", data_files=os.path.join(json_dir, "hf-train-v1.1.json"), field="data", split="train")
val_data = datasets.load_dataset("json", data_files=os.path.join(json_dir, "hf-test-v1.1.json"), field="data", split="train")

train_data = train_data.map(
    process_squad_answers,
    batched=False,
)
val_data = val_data.map(
    process_squad_answers,
    batched=False,
)

train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["id", "title", "context", "question", "answers", "answer"],
    num_proc=32, # set to the number of CPU cores in AF node
)
val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["id", "title", "context", "question", "answers", "answer"],
    num_proc=32,
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

#model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, latent_size)
model = EncoderDecoderModel.from_pretrained(
    "/groups/gac50543/migrated_from_SFA_GPFS/asada/pretraining/wikipedia-en-bert-base-cased-plm0.50-lr5e-5/checkpoint-199820/",
)

model.config.is_vae = False
model.config.dropout_prob = 0.5

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# load bleu for validation
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
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
    bleu4_output = bleu.compute(predictions=no_rep_pred_str, references=label_str, max_order=4)
    meteor_output = meteor.compute(predictions=no_rep_pred_str, references=label_str)

    return {
        "bleu4": round(np.mean(bleu4_output["bleu"]), 4),
        "rougeL": round(np.mean(rouge_output["rougeL"]), 4),
        "meteor": round(np.mean(meteor_output["meteor"]), 4),
    }

run_name = "squad-nomaskdec-smooth0.1-from-pre"
# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir=os.path.join("~/my_data/AACL/qg/", run_name),
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    eval_steps=200,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    learning_rate=5e-05,
    weight_decay=0.1,
    num_train_epochs=20,
    overwrite_output_dir=True,
    save_total_limit=False,
    fp16=True,
    torch_compile=True,
    report_to="wandb",
    run_name=run_name,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
