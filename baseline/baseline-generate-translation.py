import datasets
import sys
from transformers import BertTokenizer, EncoderDecoderModel

bleu = datasets.load_metric("bleu")

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = EncoderDecoderModel.from_pretrained("/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/translation/baseline-translation-output/")
model.to("cuda")

def extract_features(examples):
    return {
        "en": [example["en"] for example in examples['translation']],
        "de": [example["de"] for example in examples['translation']],
     }

test_data = datasets.load_dataset("wmt14", "de-en", split="test[:20%]")
test_data = test_data.map(extract_features, batched=True, remove_columns=["translation"])


# only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
# test_data = test_data.select(range(16))

batch_size = 16  # change to 64 for full evaluation

# map data correctly
def generate_translation(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    # inputs = tokenizer(batch["en"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    inputs = tokenizer(batch["de"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

results = test_data.map(generate_translation, batched=True, batch_size=batch_size, remove_columns=["de"])
# results = test_data.map(generate_translation, batched=True, batch_size=batch_size, remove_columns=["en"])

pred_str = results["pred"]
label_str = results["en"]
label_str = [[item.split()] for item in label_str]
pred_str = [item.split() for item in pred_str]
bleu_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)


# print(bleu_output)
sys.stderr.write(str(bleu_output["bleu"]))

for item in pred_str:
    print(" ".join(item))