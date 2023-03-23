import os
import datasets
from datasets import load_from_disk
import transformers
from transformers import DataCollatorForLanguageModeling
import numpy as np
import evaluate

from nar_transformers import BertTokenizerFast
from nar_transformers import BertConfig
from nar_transformers import EncoderDecoderConfig, EncoderDecoderModel
from nar_transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from nar_transformers import DataCollatorForPermutationLanguageModeling

import wandb
os.environ["WANDB_PROJECT"] = "pre-training"


cached_data_dir = "/scratch/aae15163zd/cache/wikipedia-20220301en-bert-base-cased-512/"
model_name = "bert-base-cased"

if cached_data_dir is not None:
    with open(os.path.join(cached_data_dir, "model_name.txt"), "r") as f:
        assert model_name == f.read()

batch_size = 20
max_length = 512
latent_size = 8
pretraining_strategy = "mlm" # ae: AutoEncoding, mlm: Masked Language Modeling, plm: Permutation Language Modeling

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

def process_permutation_language_model_inputs(batch):
    plm_probability = 1/6

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    bs = len(input_ids)
    seq_length = attention_mask.sum(-1) - 2 # Excluding [CLS] and [SEP] tokens
    num_permuted_tokens = (seq_length * plm_probability).int()

    permuted_input_ids = input_ids.clone()
    for i in range(bs):
        target_indices = np.random.choice(np.arange(1, seq_length.numpy()[i] - 1), num_permuted_tokens.numpy()[i])
        permuted_indices = np.random.permutation(target_indices)
        permuted_input_ids[i, target_indices] = input_ids[i, permuted_indices]
    batch["input_ids"] = permuted_input_ids

    return batch

if cached_data_dir is None:
    train_data = datasets.load_dataset("wikipedia", "20220301.en", split="train[:99%]")
    val_data = datasets.load_dataset("wikipedia", "20220301.en", split="train[-1%:]")

    train_data = train_data.map(
        process_wiki_to_model_inputs,
        batched=True,
        batch_size=1024,
        remove_columns=["text"],
        num_proc=32, # set to the number of CPU cores in AF node
    )
    val_data = val_data.map(
        process_wiki_to_model_inputs,
        batched=True,
        batch_size=1024,
        remove_columns=["text"],
        num_proc=32, # set to the number of CPU cores in AF node
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
else:
    train_data = load_from_disk(os.path.join(cached_data_dir, "train"))
    val_data = load_from_disk(os.path.join(cached_data_dir, "valid"))

# Permutation lanaguage modeling
if pretraining_strategy == "plm":
    train_data = train_data.map(
        process_permutation_language_model_inputs,
        batched=True,
        batch_size=1024,
        num_proc=32, # set to the number of CPU cores in AF node
    )
    val_data = val_data.map(
        process_permutation_language_model_inputs,
        batched=True,
        batch_size=1024,
        num_proc=32, # set to the number of CPU cores in AF node
    )

val_data = val_data.select(range(1000))

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, latent_size)
model.config.is_vae = False

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
    output_dir="~/my_data/pretraining/wikipedia-en-bert-base-cased-novae-mlm",
    evaluation_strategy="steps",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=1_000,  # set to 1000 for full training
    save_steps=10_000,  # set to 500 for full training
    eval_steps=1_000,  # set to 8000 for full training
    warmup_steps=10_000,  # set to 2000 for full training
    learning_rate=1e-04,
    weight_decay=0.01,
    num_train_epochs=10.0, # seems like the default is only 3.0
    #max_steps=300_000,
    overwrite_output_dir=True,
    save_total_limit=None,
    fp16=True,
    report_to="wandb",
    run_name="wikipedia-en-bert-base-cased-novae-mlm",
)

if pretraining_strategy == "mlm":
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
else:
    data_collator = None

def compute_metrics(pred):
    return {"None": 0.0}

# instantiate trainer
trainer = Seq2SeqTrainer(
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
