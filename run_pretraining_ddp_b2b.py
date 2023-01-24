from transformers import (BertTokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from our_transformers import EncoderDecoderModel, EncoderVaeDecoderModel
import datasets
import argparse
import logging
import random
#import evaluate
import numpy as np
from functools import partial
import torch
torch.cuda.empty_cache()
import os

logger = logging.getLogger(__name__)

#tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
encoder_max_length=512
decoder_max_length=512


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

"""
{'de': ['Wie Sie feststellen konnten, ist der gefürchtete "Millenium-Bug " nicht eingetreten. Doch sind Bürger einiger unserer Mitgliedstaaten Opfer von schrecklichen Naturkatastrophen geworden.'], 'en': ["Although, as you will have seen, the dreaded 'millennium bug' failed to materialise, still the people in a number of countries suffered a series of natural disasters that truly were dreadful."]}
"""

def process_data_to_model_inputs(batch):
    """
    {'translation': [
        [{
            'de': 'Wiederaufnahme der Sitzungsperiode',
            'en': 'Resumption of the session'
        },
        {
            'de': 'Ich bitte Sie, sich zu einer Schweigeminute zu erheben.',
            'en': "Please rise, then, for this minute' s silence."
        },
        {
            'de': 'Frau Präsidentin, zur Geschäftsordnung.',
            'en': 'Madam President, on a point of order.'
        }]
    ]}
    """
    inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=encoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = inputs.input_ids
    batch["decoder_attention_mask"] = inputs.attention_mask
    batch["labels"] = inputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
                       batch["labels"]]
    return batch

#extract the translations as columns because the format in huggingface datasets for wmt14 is not practical
def extract_features_wiki(examples):
    return {
        "text": examples["text"],
     }

def extract_features(examples):
    return {
        "en": [example["en"] for example in examples['translation']],
        "de": [example["de"] for example in examples['translation']],
     }

def manage_dataset_to_specify_bert(dataset, encoder_max_length=512, decoder_max_length=512, batch_size=1):
    bert_wants_to_see = ["input_ids", "attention_mask", "decoder_input_ids",
                         "decoder_attention_mask", "labels"]

    dataset = dataset.map(process_data_to_model_inputs,
                          batched=True,
                          batch_size=batch_size
                          )
    dataset.set_format(type="torch", columns=bert_wants_to_see)
    return dataset

# load bleu for validation
#bleu = evaluate.load("bleu")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    bleu_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)
    return {"bleu4": round(np.mean(bleu_output["bleu"]), 4)}


def main():
    parser = argparse.ArgumentParser()

    ## Batch size
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per device")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=768, type=int, help="Latent space dimension.")

    ## IO: Logging and Saving
    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=500,
                        help="Adjust save_steps for last steps to save more frequently.")
    parser.add_argument('--seed', type=int, default=99,
                        help="random seed for initialization")

    # Training Schedule
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    # Decoder Option
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--min_length", type=int, default=55)

    # Precision & Distributed Training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    #""" Start Configuring for DDP """
    args = parser.parse_args()
    set_seed(args)

    """ Initializing EncoderDecoder Model """
    EncoderVaeDecoderModel.is_nar=True
    #EncoderVaeDecoderModel.is_nar=False
    model = EncoderVaeDecoderModel.from_encoder_vae_decoder_pretrained("bert-base-cased", "bert-base-cased")
    #model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "bert-base-cased")

    """ Tokenization Part """
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    """ Model Configuration to Seq2Seq Model """
    # set special tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.unk_token_id = tokenizer.unk_token_id
    model.config.is_decoder = False
    model.config.add_cross_attention = True
    model.config.decoder.add_cross_attention = True
    model.config.is_encoder_vae_decoder = True
    model.config.is_encoder_decoder = False
    model.config.latent_size = args.latent_size
    model.config.is_nar=True
    model.config.tie_encoder_decoder=True
    model.config.output_attentions=True

    # sensible parameters for beam search
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = args.max_length
    model.config.min_length = args.min_length
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 1

    """ Load Huggingface Data """
    batch_size = args.batch_size

    #train_data = datasets.load_dataset("wikipedia", "20200501.en", split="train[:10%]")
    train_data = datasets.load_dataset("wikipedia", "20220301.simple", split="train[:10%]")
    val_data = datasets.load_dataset("wikipedia", "20220301.simple", split="train[-10%:]")

    train_data = train_data.map(extract_features_wiki, batched=True, remove_columns=["title"])
    val_data = val_data.map(extract_features_wiki, batched=True, remove_columns=["title"])

    train_data = manage_dataset_to_specify_bert(train_data, batch_size=batch_size)
    val_data = manage_dataset_to_specify_bert(val_data, batch_size=batch_size)

    """ Model Train """

    # set training arguments - these params are not really tuned, feel free to change
    training_args = Seq2SeqTrainingArguments(
        #output_dir="/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/translation/BERT2BERT-output",
        #output_dir="/media/sohrab/External Drive/BERT2BERT/results/translation/en-de/",
        output_dir="./output_dir",
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=False,
        logging_steps=2,  # set to 1000 for full training
        save_steps=100,  # set to 500 for full training
        eval_steps=10,  # set to 8000 for full training
        warmup_steps=1,  # set to 2000 for full training
        max_steps=100,  # delete for full training
        overwrite_output_dir=True,
        save_total_limit=1,
        fp16=False,
    )


    # trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        #compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    trainer.train()


if __name__ == "__main__":
    main()
