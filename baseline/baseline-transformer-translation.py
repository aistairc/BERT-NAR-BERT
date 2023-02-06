import datasets
import transformers
import numpy as np
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig
from transformers import AutoTokenizer, MarianMTModel

batch_size=16  # change to 16 for full training

#extract the translations as columns because the format in huggingface datasets for wmt14 is not practical
def extract_features(examples):
    return {
        "en": [example["en"] for example in examples['translation']],
        "de": [example["de"] for example in examples['translation']],
     }

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch['de'], text_target=batch['en'], padding='max_length', truncation=True, max_length=128)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["labels"] = inputs.labels
  return batch

train_data = datasets.load_dataset("wmt14", "de-en", split="train")
val_data = datasets.load_dataset("wmt14", "de-en", split="validation")

train_data = train_data.map(extract_features, batched=True, remove_columns=["translation"])
val_data = val_data.map(extract_features, batched=True, remove_columns=["translation"])

# Need to check how to train and pass on a custom sentencepiece model/vocabulary
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")                                                   

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

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["en", "de",]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["en", "de",]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)

config = AutoConfig.from_pretrained(
    "Helsinki-NLP/opus-mt-de-en",
    vocab_size=len(tokenizer),
    n_ctx=128,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


model = MarianMTModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Marian MT model size: {model_size/1000**2:.1f}M parameters")

    
# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/translation/baseline-translation-output-marian-mt-full-cont/",
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=1000,  # set to 1000 for full training
    save_steps=2000,  # set to 500 for full training
    eval_steps=2000,  # set to 8000 for full training
    warmup_steps=1000,  # set to 2000 for full training
    overwrite_output_dir=True,
    save_total_limit=1,
    generation_max_length=128,
    num_train_epochs=100.0,
    predict_with_generate=True,
    fp16=True,
)

# trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()