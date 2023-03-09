import os
import datasets
import transformers
from transformers import DataCollatorForLanguageModeling, DataCollatorForPermutationLanguageModeling
import numpy as np
import evaluate

from nar_transformers import BertTokenizerFast
from nar_transformers import BertConfig
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import wandb
os.environ["WANDB_PROJECT"] = "pre-training"


model_name = "bert-base-cased"
batch_size = 16  #
max_length = 512 #
latent_size = 8

tokenizer = BertTokenizerFast.from_pretrained(model_name)

def process_wiki_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = inputs.input_ids
    batch["decoder_attention_mask"] = inputs.attention_mask
    batch["labels"] = inputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

all_data = datasets.load_dataset("wikipedia", "20220301.en", split="train")

all_data = all_data.map(
    process_wiki_to_model_inputs,
    batched=True,
    batch_size=1024,
    remove_columns=["text"],
    num_proc=32, # set to the number of CPU cores in AF node
)

train_data = all_data.select(range(len(all_data)))
val_data = all_data.select(range(1000))

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

encoder_config = BertConfig.from_pretrained(model_name)
decoder_config = BertConfig(**encoder_config.to_dict())
encoder_config.latent_size, decoder_config.latent_size = latent_size, latent_size
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

model = EncoderDecoderModel(config=config)
model.config.is_vae = True

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.config.vocab_size = model.config.decoder.vocab_size

# sensible parameters for beam search
# actually, these parameters are not used for NAR model
model.config.max_length = max_length + 1
model.config.min_length = 1
model.config.no_repeat_ngram_size = 0
model.config.early_stopping = True
model.config.length_penalty = 1.0
model.config.num_beams = 1
model.config.num_beam_groups = 0

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="~/my_data/pretraining/wiki-en-MLM",
    evaluation_strategy="no",
    save_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=1_000,  # set to 1000 for full training
    save_steps=10_000,  # set to 500 for full training
    eval_steps=1_000,  # set to 8000 for full training
    warmup_steps=10_000,  # set to 2000 for full training
    learning_rate=1e-04,
    #num_train_epochs=1.0, # seems like the default is only 3.0
    max_steps=300_000,
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True,
    report_to="wandb",
    run_name="wiki-en-MLM",
)

mlm_data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=mlm_data_collator,
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)
