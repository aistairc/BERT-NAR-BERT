#from transformers import (BertTokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from transformers import BertTokenizerFast
from our_transformers import EncoderDecoderModel, EncoderVaeDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from our_transformers.optimization import AdamW, RecAdam, anneal_function, Lamb, WarmupLinearSchedule
import datasets
import argparse
import logging
import random
import evaluate
import numpy as np
from functools import partial
from tqdm import tqdm, trange
import torch
#from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()
import os

logger = logging.getLogger(__name__)

#tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
encoder_max_length = 512
decoder_max_length = 512


def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('WORLD_SIZE') or 1)


def ompi_local_rank():
    """Find OMPI local rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('LOCAL_RANK') or 0)


def ompi_local_size():
    """Find OMPI local size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('NUM_GPUS_PER_NODE') or 1)


def get_master_ip(master_name=None):
    return os.environ.get('MASTER_ADDR', '127.0.0.1')


def get_master_port():
    return os.environ.get('MASTER_PORT', '29500')


def get_num_gpus():
    return int(os.environ.get('NUM_GPUS_PER_NODE', 1))


def gpu_indices(divisible=True):
    """Get the GPU device indices for this process/rank
    :param divisible: if GPU count of all ranks must be the same
    :rtype: list[int]
    """
    local_size = ompi_local_size()
    local_rank = ompi_local_rank()
    assert 0 <= local_rank < local_size, "Invalid local_rank: {} local_size: {}".format(local_rank, local_size)
    gpu_count = get_num_gpus()
    assert gpu_count >= local_size > 0, "GPU count: {} must be >= LOCAL_SIZE: {} > 0".format(gpu_count, local_size)
    if divisible:
        ngpu = gpu_count / local_size
        gpus = np.arange(local_rank * ngpu, (local_rank + 1) * ngpu)
        if gpu_count % local_size != 0:
            logger.warning(
                "gpu_count: {} not divisible by local_size: {}; some GPUs may be unused".format(gpu_count, local_size))
    else:
        gpus = np.array_split(range(gpu_count), local_size)[local_rank]

    ret_gpus = [int(g) for g in gpus]
    return ret_gpus


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
bleu = evaluate.load("bleu")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    bleu_output = bleu.compute(predictions=pred_str, references=label_str, max_order=4)
    return {"bleu4": round(np.mean(bleu_output["bleu"]), 4)}


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.5, ratio_zero=0.3):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else:
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L


def train(args, train_data, model, tokenizer):
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    """
    print(train_data)

    args.n_gpu = (ompi_size() if args.local_rank != -1 else 1)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_data) // args.gradient_accumulation_steps) + 1
    else:
        #t_total = (len(train_dataloader)  * num_files) // args.gradient_accumulation_steps  # * args.num_train_epochs
        t_total = len(train_data) // args.gradient_accumulation_steps  # * args.num_train_epochs

    if args.distributed:
        t_total = t_total // ompi_size()

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    logger.info("Use AdamW optimizer.")
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        # if args.n_gpu > 1:
        #     model_vae = torch.nn.DataParallel(model_vae, device_ids=range(args.n_gpu)).to(args.device)

        gpus = list(gpu_indices())
        if args.distributed:
            #model_vae = torch.nn.parallel.DistributedDataParallel(model_vae, device_ids=gpus)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus, find_unused_parameters=True)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)  # .to(args.device)

    # Train!
    logger.info("***** Running training *****")
    #logger.info("  Num examples of first file = %d", train_data.num_examples)
    logger.info("  Num examples of first file = %d", len(train_data[1]))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    ompi_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    dist_sum, dist_num = 0.0, 0
    model.zero_grad()
    num_train_epochs_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    n_iter_per_file = len(train_data[1]) / args.train_batch_size
    n_iter = int(args.num_train_epochs * (len(train_data[1]) // args.train_batch_size))
    #n_iter = int(args.num_train_epochs * n_iter_per_file * num_files)

    if args.transition_learning == "ae":
        beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=0.0, n_cycle=10,
                                               ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)

    elif args.transition_learning == "vae":
        beta_t_list = frange_cycle_zero_linear(n_iter, start=0.1, stop=args.beta, n_cycle=10,
                                               ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)

    elif args.transition_learning == "ae2vae":
        beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta, n_cycle=10,
                                               ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)

    logger.info(
        f"Total iters (estimated): {n_iter}; Length of beta schedule: {len(beta_t_list)}; #Iter per file {n_iter_per_file}")


    beta_t = 0.0

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in range(int(args.num_train_epochs)):  # num_train_epochs_iterator:
        print("epoch: ", epoch)
        #train_data.reset()

        for step, batch in enumerate(train_data):
            """
            if args.generation_model_type == "nag":
                if args.use_different_model_type:
                    tokenized_text0, tokenized_text1, tokenized_text_lengths = batch
                else:
                    tokenized_text0, tokenized_text_lengths = batch
                    tokenized_text1 = tokenized_text0.clone()

            inputs = tokenized_text0
            labels = tokenized_text1
            tokenized_text1 = tokenized_text1.to(args.device)
            """

            inputs = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            decoder_input_ids = batch["decoder_input_ids"]
            decoder_attention_mask = batch["decoder_attention_mask"]
            labels = batch["labels"]

            """ For GPU Training """
            #inputs = inputs.to(args.device)
            #labels = labels.to(args.device)

            model.train()

            if args.use_beta_schedule:
                if global_step >= len(beta_t_list):
                    beta_t = 1.0
                else:
                    beta_t = beta_t_list[global_step]

            model.config.args.beta = beta_t

            if beta_t == 0.0:
                model.config.args.fb_mode = 0
            else:
                model.config.args.fb_mode = 1

            if args.use_deterministic_connect:
                model.config.args.fb_mode = 2

            #loss_rec, loss_kl, loss, encoder_time, decoder_time = model(inputs, labels)
            loss_rec, loss_kl, loss = model(inputs,
                                            attention_mask,
                                            decoder_input_ids,
                                            decoder_attention_mask,
                                            labels)

            loss_rec = loss_rec.mean()  # mean() to average on multi-gpu parallel training
            loss_kl = loss_kl.mean()
            loss = loss.mean()

            if args.use_philly:
                # if args.local_rank in [-1, 0]:
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("Steps {}, Rank {}, Epoch: [{}/{}][{}/{}], Beta: {}, Loss: {}, kl_Loss: {}, ppl: {}".
                                format(global_step, ompi_rank(), epoch, args.num_train_epochs,
                                       step, n_iter_per_file, model.module.args.beta, loss_rec, loss_kl,
                                       torch.exp(loss_rec.clone().detach())))
                    logger.info("PROGRESS: {}%".format(round(100 * global_step * args.gradient_accumulation_steps / n_iter, 4)))
                    logger.info("EVALERR: {}%".format(loss_rec))
                    logger.info("Learning_Rate: {}".format(scheduler.get_last_lr()[0]))


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                """ Added for RecAdam Optimizer"""
                if args.logging_Euclid_dist:
                    dist = torch.sum(torch.abs(torch.cat(
                        [p.view(-1) for n, p in model.named_parameters() if args.encoder_model_type in n]) - torch.cat(
                        [p.view(-1) for n, p in pretrained_model.named_parameters() if
                         args.encoder_model_type in n])) ** 2).item()
                else:
                    dist = 0.0
                dist_sum += dist
                dist_num += 1

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                # do not use scheduler for lamb
                # if not args.use_lamb_optim:
                scheduler.step()  # Update learning rate schedule
                #scheduler.get_last_lr()
                model.zero_grad()

                global_step += 1
                args.global_steps = global_step
                if global_step % 100 == 0:
                    logger.info("Global Steps: {}".format(global_step))

                # adjust save_steps to save more frequently updates at last steps
                if global_step >= args.save_new_steps_from:
                    args.save_steps = args.save_new_steps

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    """ Added for RecAdam Optimizer"""
                    anneal_lambda = anneal_function(args.recadam_anneal_fun, global_step, args.recadam_anneal_k,
                                                    args.recadam_anneal_t0, args.recadam_anneal_w)
                    if args.use_RecAdam_optim:
                        logger.info(
                            "Epoch: {}, Step: {}, Training loss: {:.2e} (avg {:.2e}), Euclid distance: {:.2e} (avg {:.2e}),"
                            "anneal lambda: {:.2e}, lr: {:.2e}".format(
                                epoch, step, loss.item(), (tr_loss - logging_loss) / args.logging_steps,
                                dist, dist_sum / dist_num, anneal_lambda, scheduler.get_last_lr()[0]))
                    else:
                        logger.info(
                            "Epoch: {}, Step: {}, Training loss: {:.2e} (avg {:.2e}), lr: {:.2e}".format(
                                epoch, step, loss.item(), (tr_loss - logging_loss) / args.logging_steps, scheduler.get_last_lr()[0]))
                    #logging_loss = tr_loss
                    #dist_sum, dist_num = 0.0, 0

                    # Log metrics
                    """
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar('ppl', torch.exp(torch.tensor(tr_loss - logging_loss) / args.logging_steps))
                    """
                    logging_loss = tr_loss
                    dist_sum, dist_num = 0.0, 0

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    print("Save Checkpoints: ", args.save_steps)
                    save_checkpoint(model, optimizer, global_step, args)

            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break
    """
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    """

    return global_step, tr_loss / global_step, optimizer

def main():
    parser = argparse.ArgumentParser()
    """ IO """
    parser.add_argument("--output_dir", default="./output_dir", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    """ Batch size """
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per device")
    """ Variational auto-encoder """
    parser.add_argument("--latent_size", default=768, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true',
                        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--use_beta_schedule", action='store_true',
                        help="Use cyclical beta schedule for auto-encoders.")
    """ IO: Logging and Saving """
    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=500,
                        help="Adjust save_steps for last steps to save more frequently.")
    parser.add_argument('--seed', type=int, default=99,
                        help="random seed for initialization")
    """ Training Schedules """
    parser.add_argument("--ratio_increase", default=0.25, type=float,
                        help="Learning schedule, the percentage for the annealing stage.")
    parser.add_argument("--ratio_zero", default=0.25, type=float,
                        help="Learning schedule, the percentage for the pure auto-encoding stage.")
    parser.add_argument("--fb_mode", default=0, type=int,
                        help="free bit training mode.")
    parser.add_argument("--dim_target_kl", default=3.0, type=float,
                        help="dim_target_kl free bit training mode.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="If > 0: set total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    """ Decoder Option """
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--min_length", type=int, default=55)
    """ Precision & Distributed Training """
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--world-size', default=ompi_size(), type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://' + get_master_ip() + ':' + get_master_port(), type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--port', type=str, default=get_master_port(), help="Port")
    """ Objective Function """
    parser.add_argument("--do_train", default="huggingface_trainer", type=str,
                        help="huggingface_trainer / custom_trainer")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--vae", action='store_true',
                        help="Whether to run vae model over the BERT-to-BERT.")
    parser.add_argument("--loading_dataset", default="huggingface", type=str,
                        help="huggingface / pre-processed ")
    parser.add_argument("--transition_learning", default="ae2vae", type=str,
                        help="ae / vae / ae2vae")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="The weighting hyper-parameter of the KL term in VAE")
    args = parser.parse_args()
    set_seed(args)
    args.dist_url = 'tcp://' + get_master_ip() + ':' + args.port
    rank_node = ompi_rank()
    args.distributed = args.world_size >= 1
    logger.info("Rank {} distributed: {}".format(rank_node, args.distributed))

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    """ Setup distant debugging if needed """
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    """
    if args.distributed:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=ompi_rank(),
            group_name='mtorch')
    logger.info(
        "World Size is {}, Backend is {}, Init Method is {}, rank is {}".format(args.world_size, args.dist_backend,
                                                                                args.dist_url, ompi_rank()))
    """
    """
    gpus = list(gpu_indices())
    args.n_gpu = len(gpus)
    args.local_rank = ompi_rank()  # gpus[0]
    torch.cuda.set_device(gpus[0])
    device = torch.device("cuda", gpus[0])

    args.device = device
    logger.info('Rank {}, gpus: {}, get_rank: {}'.format(rank_node, gpus, torch.distributed.get_rank()))
    logger.info(f'Local rank is {args.local_rank}, {rank_node}')

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    """

    """ Initializing EncoderDecoder Model """
    EncoderVaeDecoderModel.is_nar = True
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
    model.config.is_nar = True
    model.config.tie_encoder_decoder = True
    model.config.output_attentions = True

    """ sensible parameters for beam search """
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = args.max_length
    model.config.min_length = args.min_length
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 1
    model.config.args = args

    """ Load Huggingface Data """
    batch_size = args.batch_size
    if args.loading_dataset == "huggingface":
        #train_data = datasets.load_dataset("wikipedia", "20200501.en", split="train[:10%]")
        train_data = datasets.load_dataset("wikipedia", "20220301.simple", split="train[:1%]")
        val_data = datasets.load_dataset("wikipedia", "20220301.simple", split="train[-1%:]")

        train_data = train_data.select(range(5))
        val_data = val_data.select(range(5))

        train_data = train_data.map(extract_features_wiki, batched=True, remove_columns=["title"])
        val_data = val_data.map(extract_features_wiki, batched=True, remove_columns=["title"])

        train_data = manage_dataset_to_specify_bert(train_data, batch_size=batch_size)
        val_data = manage_dataset_to_specify_bert(val_data, batch_size=batch_size)

    elif args.loading_dataset == "pre-processed":
        pass

    if args.do_train == "huggingface_trainer":
        """ set training arguments - these params are not really tuned, feel free to change """
        training_args = Seq2SeqTrainingArguments(
            output_dir="./output_dir",
            evaluation_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=False,
            logging_steps=args.logging_steps,  # set to 1000 for full training
            save_steps=args.save_steps,  # set to 500 for full training
            eval_steps=args.eval_steps,  # set to 8000 for full training
            warmup_steps=1,  # set to 2000 for full training
            num_train_epochs=args.num_train_epochs,  # delete for full training
            overwrite_output_dir=True,
            save_total_limit=1,
            fp16=args.fp16,
        )
        """ trainer """
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            #compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset=val_data,
        )

        trainer.train()

    elif args.do_train == "custom_trainer":
        global_step, tr_loss, optimizer = train(args, train_data, model, tokenizer)
        print("global_step: {}, tr_loss: {}, optimizer: {}".format(global_step, tr_loss, optimizer))
        #print("global_step: {}, tr_loss: {}, optimizer: {}, encoder_time: {}, decoder_time: {}".format(
        #    global_step, tr_loss, optimizer, encoder_time, decoder_time))
        """
        set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
        for epoch in range(int(args.num_train_epochs)):  # num_train_epochs_iterator:
            print("epoch: ", epoch)
            #train_data.reset()
            for step, batch in enumerate(train_data):
                print(step)
                print(batch)
                exit()
        """
    if args.do_eval:
        pass


if __name__ == "__main__":
    main()
