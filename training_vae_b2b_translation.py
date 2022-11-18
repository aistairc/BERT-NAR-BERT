from pytorch_transformers import (EncoderVaeDecoderModel, BertTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer)
#from pytorch_transformers import (EncoderDecoderModel, BertTokenizer, TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer)
import datasets
from functools import partial
import torch
torch.cuda.empty_cache()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def process_data_to_model_inputs(batch, encoder_max_length=512, decoder_max_length=512, batch_size=1):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    #print(batch)
    #exit()
    """
    {'translation': [[{'de': 'Wiederaufnahme der Sitzungsperiode', 'en': 'Resumption of the session'}, {'de': 'Ich erkläre die am Freitag, dem 17. Dezember unterbrochene Sitzungsperiode des Europäischen Parlaments für wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum Jahreswechsel und hoffe, daß Sie schöne Ferien hatten.', 'en': 'I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.'}, {'de': 'Wie Sie feststellen konnten, ist der gefürchtete "Millenium-Bug " nicht eingetreten. Doch sind Bürger einiger unserer Mitgliedstaaten Opfer von schrecklichen Naturkatastrophen geworden.', 'en': "Although, as you will have seen, the dreaded 'millennium bug' failed to materialise, still the people in a number of countries suffered a series of natural disasters that truly were dreadful."}, {'de': 'Im Parlament besteht der Wunsch nach einer Aussprache im Verlauf dieser Sitzungsperiode in den nächsten Tagen.', 'en': 'You have requested a debate on this subject in the course of the next few days, during this part-session.'}, {'de': 'Heute möchte ich Sie bitten - das ist auch der Wunsch einiger Kolleginnen und Kollegen -, allen Opfern der Stürme, insbesondere in den verschiedenen Ländern der Europäischen Union, in einer Schweigeminute zu gedenken.', 'en': "In the meantime, I should like to observe a minute' s silence, as a number of Members have requested, on behalf of all the victims concerned, particularly those of the terrible storms, in the various countries of the European Union."}, {'de': 'Ich bitte Sie, sich zu einer Schweigeminute zu erheben.', 'en': "Please rise, then, for this minute' s silence."}, {'de': '(Das Parlament erhebt sich zu einer Schweigeminute.)', 'en': "(The House rose and observed a minute' s silence)"}, {'de': 'Frau Präsidentin, zur Geschäftsordnung.', 'en': 'Madam President, on a point of order.'}, {'de': 'Wie Sie sicher aus der Presse und dem Fernsehen wissen, gab es in Sri Lanka mehrere Bombenexplosionen mit zahlreichen Toten.', 'en': 'You will be aware from the press and television that there have been a number of bomb explosions and killings in Sri Lanka.'}, {'de': 'Frau Präsidentin, zur Geschäftsordnung.', 'en': 'Madam President, on a point of order.'}]]}
    """
    inputs = tokenizer([segment["en"] for segment in batch['translation']],
                       padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer([segment["de"] for segment in batch['translation']],
                        padding="max_length", truncation=True, max_length=encoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
                       batch["labels"]]
    return batch


def manage_dataset_to_specify_bert(dataset, encoder_max_length=512, decoder_max_length=512, batch_size=1):
    bert_wants_to_see = ["input_ids", "attention_mask", "decoder_input_ids",
                         "decoder_attention_mask", "labels"]

    _process_data_to_model_inputs = partial(process_data_to_model_inputs,
                                            encoder_max_length=encoder_max_length,
                                            decoder_max_length=decoder_max_length,
                                            batch_size=batch_size
                                            )
    dataset = dataset.map(_process_data_to_model_inputs,
                          batched=True,
                          batch_size=batch_size
                          )
    dataset.set_format(type="torch", columns=bert_wants_to_see)
    return dataset


model = EncoderVaeDecoderModel.from_encoder_vae_decoder_pretrained("bert-base-multilingual-uncased",
                                                                   "bert-base-multilingual-uncased")

""" Tokenization Part """

# tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

""" Model Configuration to Seq2Seq Model """

# set special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.is_decoder = True
model.config.add_cross_attention = True
model.config.decoder.add_cross_attention = True
model.config.is_encoder_vae_decoder = True
model.config.is_encoder_decoder = False
model.config.latent_size = 768

# sensible parameters for beam search
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.length_penalty = 2.0
model.config.num_beams = 1

""" Get The data """

train_data = datasets.load_dataset("wmt14", "de-en", split="train")
val_data = datasets.load_dataset("wmt14", "de-en", split="validation[:10%]")

train_data = train_data.select(range(10))
val_data = val_data.select(range(10))

train_data = manage_dataset_to_specify_bert(train_data)
val_data = manage_dataset_to_specify_bert(val_data)

# train_data = train_data.map(process_data_to_model_inputs, batch_size=batch_size, batched=True)

# train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])

""" Model Train """

batch_size = 1
# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
#training_args = TrainingArguments(
    output_dir="./results/translation/vae-en-de",
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
    # fp16=True,
)

# trainer
trainer = Seq2SeqTrainer(
#trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()