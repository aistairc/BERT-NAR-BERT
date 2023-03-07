import os
import datasets
import transformers
import numpy as np
import evaluate
from nar_transformers import BertTokenizerFast
from nar_transformers import BertConfig
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel, EncoderDecoderPosteriorModel
from nar_transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import wandb


tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

batch_size=32  # change to 16 for full training
max_length=128 # 128 actually works better for MT

latent_size = 8

#extract the translations as columns because the format in huggingface datasets for wmt14 is not practical
def extract_features(examples):
    return {
        "en": [example["en"] for example in examples['translation']],
        "de": [example["de"] for example in examples['translation']],
    }

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["en"], padding="max_length", truncation=True, max_length=max_length)
    outputs = tokenizer(batch["de"], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch


dataset_cache_path = "/scratch/aae15163zd/cache/huggingface/datasets/wmt14-en-de-msl128-multilingual-bert-base-cased/"

if False:
    train_data = datasets.load_dataset("wmt14", "de-en", split="train")
    val_data = datasets.load_dataset("wmt14", "de-en", split="validation")

    train_data = train_data.map(extract_features, batched=True, remove_columns=["translation"])
    val_data = val_data.map(extract_features, batched=True, remove_columns=["translation"])

    from datasets.utils.logging import disable_progress_bar, enable_progress_bar
    disable_progress_bar()
    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["en", "de",],
        num_proc=32,
    )
    val_data = val_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["en", "de",],
        num_proc=32,
    )
    enable_progress_bar()
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    train_data.save_to_disk(os.path.join(dataset_cache_path, 'train'))
    val_data.save_to_disk(os.path.join(dataset_cache_path, 'valid'))
else:
    train_data = datasets.load_from_disk(os.path.join(dataset_cache_path, 'train'))
    val_data = datasets.load_from_disk(os.path.join(dataset_cache_path, 'valid'))

    distilled_train_data = datasets.load_from_disk(os.path.join(dataset_cache_path, 'distilled_train'))

train_data_type = "distilled"
if train_data_type == "raw":
    train_data = raw_train_data
elif train_data_type == "distilled":
    train_data = distilled_train_data
elif train_data_type == "mix":
    train_data = datasets.concatenate_datasets([raw_train_data, distilled_train_data])
else:
    raise ValueError()

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

encoder_config = BertConfig.from_pretrained("bert-base-multilingual-cased")
decoder_config = BertConfig(**encoder_config.to_dict())
encoder_config.latent_size, decoder_config.latent_size = latent_size, latent_size
encoder_config.is_vae = True
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

model = EncoderDecoderModel(config=config)

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


# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="~/my_data/translation/ctc-only-multi-vae-distilled/",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=500,  # set to 1000 for full training
    save_steps=5000,  # set to 500 for full training
    eval_steps=500,  # set to 8000 for full training
    warmup_steps=10000,  # set to 2000 for full training
    #warmup_ratio=0.10,
    learning_rate=1e-04,
    #num_train_epochs=10.0, # seems like the default is only 3.0
    max_steps=300000,
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True,
    report_to="wandb",
    run_name="foo-onlyctc-multi-vae-distilled",
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
