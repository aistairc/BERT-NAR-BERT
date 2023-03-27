import os
import datasets
import transformers
import numpy as np
import evaluate
from nar_transformers import BertTokenizerFast
from nar_transformers import BertConfig
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import wandb
os.environ["WANDB_PROJECT"] = "summarization"


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


train_data = datasets.load_dataset("xsum", split="train")
val_data = datasets.load_dataset("xsum", split="validation")

train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["document", "summary",],
    num_proc=32, # set to the number of CPU cores in AF node
)
val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["document", "summary",],
    num_proc=32,
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, latent_size)

#model = EncoderDecoderModel.from_pretrained(
#    "/groups/gac50543/migrated_from_SFA_GPFS/asada/pretraining/wikipedia-en-bert-base-cased-noval-mlm-fp32ctc/checkpoint-59946/",
#)
model.config.is_vae = False

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = max_length + 1
model.config.min_length = 1
model.config.no_repeat_ngram_size = 0
model.config.early_stopping = True
model.config.length_penalty = 1.0
model.config.num_beams = 1
model.config.num_beam_groups = 0


# load bleu for validation
rouge = evaluate.load("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Removing repetition tokens
    def remove_repetition(token_ids):
        no_repetition_token_ids = []
        for i, token_id in enumerate(token_ids):
            if i != len(token_ids) - 1:
                if token_ids[i + 1] == token_id:
                    token_id = tokenizer.pad_token_id
            no_repetition_token_ids.append(token_id)
        return no_repetition_token_ids
    no_rep_pred_ids = [remove_repetition(x) for x in pred_ids]
    no_rep_pred_str = tokenizer.batch_decode(no_rep_pred_ids, skip_special_tokens=True)

    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    #rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    rouge_output = rouge.compute(predictions=no_rep_pred_str, references=label_str)

    return {
        "rouge1": round(np.mean(rouge_output["rouge1"]), 4),
        "rouge2": round(np.mean(rouge_output["rouge2"]), 4),
        "rougeL": round(np.mean(rouge_output["rougeL"]), 4),
    }


# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="~/my_data/summarization/xsum",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=500,  # set to 1000 for full training
    save_steps=5_000,  # set to 500 for full training
    eval_steps=500,  # set to 8000 for full training
    warmup_steps=10_000,  # set to 2000 for full training
    learning_rate=1e-04,
    max_steps=100_000,
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True,
    weight_decay=0.01,
    report_to="wandb",
    run_name="xsum",
    gradient_accumulation_steps=1,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
