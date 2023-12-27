# BERT-NAR-BERT
BERT-NAR-BERT (BnB) is a pre-trained non-autoregressive sequence-to-sequence model, which employs BERT as the backbone for the encoder and decoder for natural language understanding and generation tasks. During the pre-training and fine-tuning with BERT-NAR-BERT, two challenging aspects are considered by adopting the length classification and connectionist temporal classification models to control the output length of BnB. We evaluate it using a standard natural language understanding benchmark GLUE and three generation tasks – abstractive summarization, question generation, and machine translation. Our results show substantial improvements in inference speed (on average 10x faster) with only little deficiency in output quality when compared to our direct autoregressive baseline BERT2BERT model.


Architecture
---------

![N|Solid](https://github.com/aistairc/BERT-NAR-BERT/blob/readme-update/BnB_Architecture.png?raw=true)

The S2S BERT-NAR-BERT (BnB) architecture. The + sign to the right of the encoder box indicates, the input embeddings are the sum of the token embeddings, the position embeddings, and the type embeddings where the decoder box indicates the sum of the position, the type, and the latent embeddings.


	
Examples
---------

Our scripts such as `v-job-pretraining.sh`, `v-job-summarization.sh`, and `v-job-translation.sh` were mainly prepared for launching multi-node training on the ABCI computation cluster. However, you can also just run the individual training files like `train-summarization.py` or `train-translation.py` after updating their internal training arguments and making sure everything matches your setup.

We have also prepared training scripts for our autoregressive baseline BERT-TO-BERT in the `baseline` directory, and scripts to run model evaluations on GLUE benchmark tasks in the `glue_task` directory.

		
Publications
---------

If you use this tool, please cite the following paper:

Mohammad Golam Sohrab, Masaki Asada, Matīss Rikters, Makoto Miwa (2023). "[BERT-NAR-BERT: A Non-autoregressive Pre-trained Sequence-to-Sequence Model Leveraging BERT Checkpoints](https://ieeexplore.ieee.org/document/10373869)." IEEE Access (2023).

```bibtex
@ARTICLE{Sohrab-EtAl2023IEEE,
	author = {Sohrab, Mohammad Golam and Asada, Masaki and Rikters, Matīss and Miwa, Makoto},
	journal={IEEE Access},
	volume={},
	number={},
	pages = {1--12},
	doi={10.1109/ACCESS.2023.3346952},
	title = {{BERT-NAR-BERT: A Non-autoregressive Pre-trained Sequence-to-Sequence Model Leveraging BERT Checkpoints}},
	year = {2023}
}
```

		
Acknowledgment
---------

This research is based on results obtained from a project JPNP20006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO). 
