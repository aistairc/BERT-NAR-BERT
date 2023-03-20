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
os.environ["WANDB_PROJECT"] = "translation_dev"


src_model_name = "bert-base-cased"
tgt_model_name = "bert-base-german-cased"
#src_model_name = "bert-base-multilingual-cased"
#tgt_model_name = "bert-base-multilingual-cased"

batch_size = 128  # change to 16 for full training
max_length = 128 # 128 actually works better for MT
latent_size = 8

src_tokenizer = BertTokenizerFast.from_pretrained(src_model_name)
tgt_tokenizer = BertTokenizerFast.from_pretrained(tgt_model_name)

#extract the translations as columns because the format in huggingface datasets for wmt14 is not practical
def extract_features(examples):
    return {
        "en": [example["en"] for example in examples['translation']],
        "de": [example["de"] for example in examples['translation']],
    }

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = src_tokenizer(batch["en"], padding="max_length", truncation=True, max_length=max_length)
    outputs = tgt_tokenizer(batch["de"], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tgt_tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

train_data = datasets.load_dataset("wmt14", "de-en", split="train")
val_data = datasets.load_dataset("wmt14", "de-en", split="validation")

train_data = train_data.map(extract_features, batched=True, remove_columns=["translation"])
val_data = val_data.map(extract_features, batched=True, remove_columns=["translation"])

train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["en", "de",],
    num_proc=32, # set to the number of CPU cores in AF node
)
val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["en", "de",],
    num_proc=32,
)

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
#distilled_train_data = datasets.load_from_disk("/scratch/aae15163zd/cache/huggingface/datasets/wmt14-en-de-msl128-multilingual-bert-base-cased/distilled_train/")
distilled_train_data = datasets.load_from_disk("/scratch/aae15163zd/cache/huggingface/datasets/wmt14-en-de-msl128-unilingual-bert-base-cased/distilled_train/")["train"]

src_tokenizer.bos_token = src_tokenizer.cls_token
src_tokenizer.eos_token = src_tokenizer.sep_token
tgt_tokenizer.bos_token = tgt_tokenizer.cls_token
tgt_tokenizer.eos_token = tgt_tokenizer.sep_token

encoder_config = BertConfig.from_pretrained(src_model_name)
decoder_config = BertConfig.from_pretrained(tgt_model_name)
encoder_config.latent_size, decoder_config.latent_size = latent_size, latent_size
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

model = EncoderDecoderModel(config=config)
model.config.is_vae = True
model.config.kl_threshold = 0.5

# set special tokens
model.config.decoder_start_token_id = tgt_tokenizer.bos_token_id
model.config.eos_token_id = tgt_tokenizer.eos_token_id
model.config.pad_token_id = tgt_tokenizer.pad_token_id

model.config.vocab_size = model.config.decoder.vocab_size

# sensible parameters for beam search
# actually, these parameters are not used for NAR model
model.config.max_length = max_length + 1 # set larger value than max_length
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

    # Removing repetition tokens
    def remove_repetition(token_ids):
        no_repetition_token_ids = []
        for i, token_id in enumerate(token_ids):
            if i != len(token_ids) - 1:
                if token_ids[i + 1] == token_id:
                    token_id = tgt_tokenizer.pad_token_id
            no_repetition_token_ids.append(token_id)
        return no_repetition_token_ids
    no_rep_pred_ids = [remove_repetition(x) for x in pred_ids]

    pred_str = tgt_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tgt_tokenizer.pad_token_id
    label_str = tgt_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    bleu_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)

    no_rep_pred_str = tgt_tokenizer.batch_decode(no_rep_pred_ids, skip_special_tokens=True)
    no_rep_bleu_output = bleu.compute(predictions=no_rep_pred_str, references=label_str, max_order=4)
    return {
        "bleu4": round(np.mean(no_rep_bleu_output["bleu"]), 4),
        "bleu4_w.o._post-process": round(np.mean(bleu_output["bleu"]), 4),
    }


# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="~/my_data/translation/foo",
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
    report_to="wandb",
    run_name="cls-th0.5",
    gradient_accumulation_steps=2,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    #train_dataset=train_data,
    train_dataset=distilled_train_data,
    eval_dataset=val_data,
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
