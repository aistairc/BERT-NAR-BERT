import datasets
import transformers
import numpy as np
import evaluate
from transformers import BertTokenizerFast
from transformers import EncoderDecoderConfig, EncoderDecoderModel, BertConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

train_data = datasets.load_dataset("wmt14", "de-en", split="train")
val_data = datasets.load_dataset("wmt14", "de-en", split="validation[:10%]")

#extract the translations as columns because the format in huggingface datasets for wmt14 is not practical
def extract_features(examples):
    return {
        "en": [example["en"] for example in examples['translation']],
        "de": [example["de"] for example in examples['translation']],
     }


train_data = train_data.map(extract_features, batched=True, remove_columns=["translation"])
val_data = val_data.map(extract_features, batched=True, remove_columns=["translation"])


tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token


batch_size=1  # change to 16 for full training
encoder_max_length=512
decoder_max_length=512

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  # inputs = tokenizer(batch["en"], padding="max_length", truncation=True, max_length=encoder_max_length)
  # outputs = tokenizer(batch["de"], padding="max_length", truncation=True, max_length=decoder_max_length)
  inputs = tokenizer(batch["de"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["en"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

# load bleu for validation
bleu = evaluate.load("bleu")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # label_str = [[item.split()] for item in label_str]
    # pred_str = [item.split() for item in pred_str]
    bleu_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)
    return {"bleu4": round(np.mean(bleu_output["bleu"]), 4)}
    

# only use 32 training examples for notebook - DELETE LINE FOR FULL TRAINING
train_data = train_data.select(range(100))

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["en", "de",]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
val_data = val_data.select(range(100))

val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["en", "de",]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


encoder_config = BertConfig.from_pretrained("bert-base-multilingual-uncased")
decoder_config = BertConfig(**encoder_config.to_dict())
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
model = EncoderDecoderModel(config=config)


# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = 512
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.length_penalty = 2.0
model.config.num_beams = 1
# model.config.num_beams = 4

    
# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/translation/baseline-translation-output/",
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=2,  # set to 1000 for full training
    save_steps=50,  # set to 500 for full training
    eval_steps=1,  # set to 8000 for full training
    warmup_steps=1,  # set to 2000 for full training
    max_steps=100,  # delete for full training
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True, 
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