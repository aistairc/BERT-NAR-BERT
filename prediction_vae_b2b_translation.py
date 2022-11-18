from pytorch_transformers import (BertTokenizer, EncoderVaeDecoderModel)
import datasets
from datasets import load_dataset
from functools import partial

def process_data_to_model_inputs(batch, encoder_max_length=512, decoder_max_length=512, batch_size=2):
    inputs = tokenizer([segment["en"] for segment in batch['translation']],
                       padding="max_length", truncation=True, max_length=encoder_max_length)
    #outputs = tokenizer([segment["de"] for segment in batch['translation']],
    #                    padding="max_length", truncation=True, max_length=encoder_max_length)
    references = [segment["de"] for segment in batch['translation']]


    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["references"] = references
    #batch["decoder_input_ids"] = outputs.input_ids
    #batch["decoder_attention_mask"] = outputs.attention_mask
    #batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    #batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
    #                   batch["labels"]]
    return batch

def manage_dataset_to_specify_bert(dataset, encoder_max_length=512, decoder_max_length=512, batch_size=10):
    #bert_wants_to_see = ["input_ids", "attention_mask", "decoder_input_ids",
    #                     "decoder_attention_mask", "labels"]
    bert_wants_to_see = ["input_ids", "attention_mask"]

    _process_data_to_model_inputs = partial(process_data_to_model_inputs,
                                            encoder_max_length=encoder_max_length,
                                            decoder_max_length=decoder_max_length,
                                            batch_size=batch_size
                                            )
    dataset = dataset.map(_process_data_to_model_inputs,
                          batched=True,
                          batch_size=batch_size
                          )
    #print(dataset)
    #exit()
    dataset.set_format(type="torch", columns=bert_wants_to_see)
    return dataset

bert2bert = EncoderVaeDecoderModel.from_pretrained('./results/translation/vae-en-de/checkpoint-100')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

test_data = datasets.load_dataset("wmt14", "de-en", split="train")
test_data = test_data.select(range(10))
test_data = manage_dataset_to_specify_bert(test_data)

references = test_data['references']
print("references", references)

#test_data = load_dataset('csv', data_files='./livedoornews.csv')['train']
#test_data = test_data.select(range(16))
#def test_encode(examples):
#    return tokenizer(examples['body'], truncation=True, padding='max_length')

#test_data = test_data.map(test_encode, batched=True, batch_size=16)
#test_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
bert2bert.config.max_length = 103
bert2bert.config.min_length = 3
#bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.no_repeat_ngram_size = 1
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 3.0
#bert2bert.config.num_beams = 4
bert2bert.config.num_beams = 1
bert2bert.config.num_beam_groups = 1
bert2bert.config.do_sample = False
#bert2bert.config.add_cross_attention=True

bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
bert2bert.config.eos_token_id = tokenizer.eos_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

# I am making predictions in batches for memory management purposes.
predicted_str = []
batch_size = 1
n_test = len(test_data['input_ids'])
for i in range(0, 10, batch_size):
    input_ids = test_data[i:i+batch_size]['input_ids']
    attention_mask = test_data[i:i+batch_size]['attention_mask']
    encoded = bert2bert.generate(input_ids, attention_mask=attention_mask)

    predicted_str.extend(tokenizer.batch_decode(encoded, skip_special_tokens=True))

print("Prediction", predicted_str)

references_list, predicted_list = [], []
for i, (ref, pred) in enumerate(zip(references, predicted_str)):
    #print("Reference_{}: {}\tPrediction_{}:{}".format(ref, pred))
    references_list.append([ref.split(" ")])
    predicted_list.append(pred.split(" "))
    #print(ref_list)
    #exit()
    print("Reference_{}: {}".format(i, ref))
    print("Prediction_{}: {}\n".format(i, pred))

rouge = datasets.load_metric("rouge")
rouge_output = rouge.compute(predictions=predicted_str, references=references, rouge_types=["rouge2"])["rouge2"].mid

print(rouge_output)
#print(predicted_list)
#print(references_list)

bleu = datasets.load_metric("bleu")
bleu_output = bleu.compute(predictions=predicted_list, references=references_list)

print(bleu_output["bleu"])

