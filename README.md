# BERT-NAR-BERT
BERT-NAR-BERT (BnB) is a pre-trained non-autoregressive sequence-to-sequence model, which employs BERT as the backbone for the encoder and decoder for natural language understanding and generation tasks. During the pre-training and fine-tuning with BERT-NAR-BERT, two challenging aspects are considered by adopting the length classification and connectionist temporal classification models to control the output length of BnB. We evaluate it using a standard natural language understanding benchmark GLUE and three generation tasks – abstractive summarization, question generation, and machine translation. Our results show substantial improvements in inference speed (on average 10x faster) with only little deficiency in output quality when compared to our direct autoregressive baseline BERT2BERT model.


		
Architecture
---------

![N|Solid](https://github.com/aistairc/BERT-NAR-BERT/blob/readme-update/BnB_Architecture.png?raw=true)

The S2S BERT-NAR-BERT (BnB) architecture. The + sign to the right of the encoder box indicates, the input embeddings are the sum of the token embeddings, the position embeddings, and the type embeddings where the decoder box indicates the sum of the position, the type, and the latent embeddings.

		
Publications
---------

If you use this tool, please cite the following paper:

Mohammad Golam Sohrab, Masaki Asada, Matīss Rikters, Makoto Miwa (2023). "BERT-NAR-BERT: A Non-autoregressive Pre-trained Sequence-to-Sequence Model Leveraging BERT Checkpoints." IEEE Access (2023).

```bibtex
@ARTICLE{Sohrab-EtAl2023IEEE,
	author = {Sohrab, Mohammad Golam and Asada, Masaki and Rikters, Matīss and Miwa, Makoto},
	journal={IEEE Access},
	volume={},
	number={},
	pages = {1--12},
	title = {{BERT-NAR-BERT: A Non-autoregressive Pre-trained Sequence-to-Sequence Model Leveraging BERT Checkpoints}},
	year = {2023}
}
```
