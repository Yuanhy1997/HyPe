import logging
import torch
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pickle
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from hype_modeling_bert import BertForSequenceClassification_HyPe
from hype_modeling_roberta import RobertaForSequenceClassification_HyPe
from hype_modeling_electra import ElectraForSequenceClassification_HyPe
from hype_modeling_xlnet import XLNetForSequenceClassification_HyPe


def noise_tune(model, noise_lambda):
    for name, params in model.named_parameters():
        if sum(params.data.shape) <= 1:
            continue
        model.state_dict()[name][:] += (torch.rand(params.size())-0.5)*noise_lambda*torch.std(params)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

xlnet_train_args = {
    "cola": (64, 1200, 120),
    "rte": (32, 800, 200),
    "mrpc": (32, 800, 200),
    "stsb": (32, 3000, 500),
}
electra_train_args = {
    "cola": (32, 3),
    "rte": (32, 10),
    "mrpc": (32, 3),
    "stsb": (32, 10),
}

logger = logging.getLogger(__name__)

def reinit_linear_head(module, std):

    # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    module.weight.data.normal_(mean=0.0, std=std)
    if module.bias is not None:
        module.bias.data.zero_()

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    low_resource: bool = field(
        default=False, 
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    checkpoint_dir: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    dropout_rate: float = field(
        default = None
    )

    # childtuning
    reserve_p: float = field(
        default=0.0
    )
    mode: str = field(
        default=None
    )

    # noisy tune
    noise_lambda: float = field(
        default=0.0
    )

    # rdrop
    rdrop_alpha: float = field(
        default=0.0
    )

    # r3f
    r3f_eps: float = field(
        default=1e-5
    )
    r3f_lambda: float = field(
        default=0.0
    )
    r3f_noise_type: str = field(
        default='normal'
    )

    # layerwise noise:
    hype_eps: float = field(
        default=1e-5
    )
    hype_type: str = field(
        default='none'
    )
    hype_top_layers: int = field(
        default=None
    )
    hype_low_layers: int = field(
        default=None
    )
    hype_intermediate: bool = field(
        default=False
    )
    hype_only_intermediate: bool = field(
        default=False
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if 'xlnet' in model_args.model_name_or_path:
        bsz, trainstep, warmstep = xlnet_train_args[data_args.task_name]
        training_args.per_device_train_batch_size = bsz
        training_args.max_steps = trainstep
        training_args.warmup_steps = warmstep
    if 'electra' in model_args.model_name_or_path:
        bsz, epo = electra_train_args[data_args.task_name]
        training_args.per_device_train_batch_size = bsz
        training_args.num_train_epochs = epo

    # Detecting last checkpoint.
    last_checkpoint = None
    outputdir_prefix = training_args.output_dir
    if not os.path.exists(outputdir_prefix):
        os.mkdir(outputdir_prefix)

    if model_args.mode is None or model_args.mode.find('Child') >= 0:
        if model_args.mode is not None:
            training_args.output_dir = training_args.output_dir+f'/task_{data_args.task_name}_seed_{training_args.seed}_lr_{training_args.learning_rate}_mode_{model_args.mode}_prob_{model_args.reserve_p}/'
        else:
            if model_args.noise_lambda > 0.:
                training_args.output_dir = training_args.output_dir+f'/task_{data_args.task_name}_seed_{training_args.seed}_lr_{training_args.learning_rate}_mode_{model_args.mode}_lam_{model_args.noise_lambda}/'
            elif model_args.rdrop_alpha > 0.:
                training_args.output_dir = training_args.output_dir+f'/task_{data_args.task_name}_seed_{training_args.seed}_lr_{training_args.learning_rate}_mode_{model_args.mode}_alpha_{model_args.rdrop_alpha}/'
            elif model_args.r3f_lambda > 0.:
                training_args.output_dir = training_args.output_dir+f'/task_{data_args.task_name}_seed_{training_args.seed}_lr_{training_args.learning_rate}_mode_{model_args.mode}_r3f_lam_{model_args.r3f_lambda}_type_{model_args.r3f_noise_type}/'
            elif model_args.hype_type != 'none':
                training_args.output_dir = training_args.output_dir+f'/task_{data_args.task_name}_seed_{training_args.seed}_lr_{training_args.learning_rate}_mode_{model_args.mode}_hype_type_{model_args.hype_type}/'
            else:
                training_args.output_dir = training_args.output_dir+f'/task_{data_args.task_name}_seed_{training_args.seed}_lr_{training_args.learning_rate}_mode_{model_args.mode}_vanilla/'

    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and skip"
            )
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue.py", data_args.task_name)
        if data_args.low_resource:
            with open(f'./1ksample_subset_{data_args.task_name}.pkl','rb') as f:
                subsample_indices = pickle.load(f)
            datasets['train'] = datasets['train'].select(subsample_indices)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.dropout_rate is not None:
        if hasattr(config, 'hidden_dropout_prob'):
            config.hidden_dropout_prob = model_args.dropout_rate
        if hasattr(config, 'dropout'):
            config.dropout = model_args.dropout_rate


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if training_args.do_train:
        model_loading_path = model_args.checkpoint_dir if model_args.checkpoint_dir else model_args.model_name_or_path
    else:
        model_loading_path = training_args.output_dir

    if model_args.hype_type != 'none':

        if 'roberta' in model_args.model_name_or_path:

            model = RobertaForSequenceClassification_HyPe.from_pretrained(
                model_loading_path,
                config=config,
            )
            model.roberta.encoder.add_noise_sampler(model_args.hype_eps, model_args.hype_type, add_intermediate= model_args.hype_intermediate, only_intermediate= model_args.hype_only_intermediate)
        
        elif 'electra' in model_args.model_name_or_path:

            model = ElectraForSequenceClassification_HyPe.from_pretrained(
                model_loading_path,
                config=config,
            )
            model.electra.encoder.add_noise_sampler(model_args.hype_eps, model_args.hype_type)

        elif 'xlnet' in model_args.model_name_or_path:

            model = XLNetForSequenceClassification_HyPe.from_pretrained(
                model_loading_path,
                config=config,
            )
            model.transformer.add_noise_sampler(model_args.hype_eps, model_args.hype_type)
        
        else:

            model = BertForSequenceClassification_HyPe.from_pretrained(
                model_loading_path,
                config=config,
            )

            # model.bert.encoder.add_noise_sampler(model_args.hype_eps, model_args.hype_type, model_args.hype_intermediate, model_args.hype_only_intermediate)
            layer_range = []
            if model_args.hype_top_layers is not None:
                layer_range += list(range(config.num_hidden_layers))[-model_args.hype_top_layers:]
            if model_args.hype_low_layers is not None:
                layer_range += list(range(config.num_hidden_layers))[:model_args.hype_low_layers]
            model.bert.encoder.add_noise_sampler(model_args.hype_eps, model_args.hype_type, model_args.hype_intermediate, model_args.hype_only_intermediate, layer_range = layer_range)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_loading_path,
            from_tf=bool(".ckpt" in model_loading_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if model_args.noise_lambda > 0:
        logger.info(f'Use noisy tuning for lambda of {model_args.noise_lambda}')
        noise_tune(model, model_args.noise_lambda)

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None or data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("metric.py", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
        
    
    logger.info('using default trainer')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                txt_name = 'seed' + training_args.output_dir.split('seed')[-1].strip('/').strip('/')
                if not training_args.do_train:
                    txt_name += '_retest'
                with open(output_eval_file, "w") as writer, open(outputdir_prefix+f'/{txt_name}.txt', 'a') as f:
                    logger.info(f"***** Eval results {task} *****")
                    f.write(f"***** Eval results {task} *****\n")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
                        f.write(f"{key} = {value}\n")
                    f.write("********************")

            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
