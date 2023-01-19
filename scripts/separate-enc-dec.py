from transformers import EncoderDecoderModel

# Path to full model for loading
model = EncoderDecoderModel.from_pretrained("/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/translation/baseline-translation-output-bigg/checkpoint-52000")
model.to("cuda")

# Paths for saving separate encoder and decoder parts
model.encoder.save_pretrained('/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/translation/baseline-translation-output-bigg/checkpoint-52000/encoder')
model.decoder.save_pretrained('/groups/3/gac50543/migrated_from_SFA_GPFS/matiss/translation/baseline-translation-output-bigg/checkpoint-52000/decoder')
