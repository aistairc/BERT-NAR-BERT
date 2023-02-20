import datasets
import transformers
import numpy as np
import evaluate
from nar_transformers import BertTokenizerFast
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel, BertConfig
from nar_transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import wandb


tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")

batch_size=16  # change to 16 for full training
max_length=128 # 128 actually works better for MT

#extract the translations as columns because the format in huggingface datasets for wmt14 is not practical
def extract_features_wiki(examples):
    return {
        "text": examples["text"],
    }

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_attention_mask"] = inputs.attention_mask
    batch["labels"] = inputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch


if False:
    train_data = datasets.load_dataset("wikipedia", "20220301.en", split="train")
    #train_data = datasets.load_dataset("wikipedia", "20220301.de", split="train")

    train_data = train_data.map(extract_features_wiki, batched=True, remove_columns=["title"])

    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        num_proc=128,
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
    )
    train_data.save_to_disk("/scratch/aae15163zd/cache/huggingface/datasets/wiki-en-full-bs16")
else:
    train_data = datasets.load_from_disk("/scratch/aae15163zd/cache/huggingface/datasets/wiki-en-full-bs16")

val_data = train_data.select(range(100))

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

encoder_config = BertConfig.from_pretrained("bert-base-multilingual-uncased")
decoder_config = BertConfig(**encoder_config.to_dict())
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

model = EncoderDecoderModel(config=config)
#model = EncoderDecoderModel.from_pretrained(
#    "/groups/gac50543/migrated_from_SFA_GPFS/asada/pretraining/VAEsent-wiki-de-full/checkpoint-20824/",
#    config=config)

model.config.do_length_prediction = True
model.config.is_vae = True
model.config.is_token_level_z = True

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = 192
model.config.min_length = 1
model.config.no_repeat_ngram_size = 0
model.config.early_stopping = True
model.config.length_penalty = 1.0
model.config.num_beams = 1
model.config.num_beam_groups = 0

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="~/my_data/pretraining/VAEsent-wiki-en-full",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=1000,  # set to 1000 for full training
    save_steps=5000,  # set to 500 for full training
    eval_steps=1000,  # set to 8000 for full training
    #warmup_steps=1000,  # set to 2000 for full training
    warmup_ratio=0.20,
    learning_rate=1e-04,
    lr_scheduler_type="constant_with_warmup",
    #max_steps=10000,  # delete for full training
    num_train_epochs=1.0, # seems like the default is only 3.0
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True,
    report_to="wandb",
    run_name="pre-VAEsent-wiki-en-full",
)

# load bleu for validation
bleu = evaluate.load("bleu")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    bleu_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)
    return {"bleu4": round(np.mean(bleu_output["bleu"]), 4)}

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    #compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
