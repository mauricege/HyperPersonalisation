"""Defines the arguments used for training and evaluation."""

import logging
from dataclasses import dataclass, field
from hyperformer.adapters import ADAPTER_CONFIG_MAPPING
from transformers import TrainingArguments
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from typing import Optional, List

arg_to_scheduler = {
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

logger = logging.getLogger(__name__)


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Contains different training parameters such as dropout, optimizers parameters, ... .
    """
    adafactor: bool = field(
        default=False, metadata={"help": "whether to use adafactor"}
    )
    dropout: Optional[float] = field(
        default=None, metadata={"help": "Dropout probability. Goes into model.config."}
    )
    attention_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Attention dropout probability. Goes into model.config."},
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={
            "help": f"Which lr scheduler to use. Selected in {sorted(arg_to_scheduler.keys())}"
        },
    )
    temperature: Optional[int] = field(
        default=1,
        metadata={
            "help": "Defines the temperature"
            "value for sampling across the multiple datasets."
        },
    )
    train_adapters: Optional[bool] = field(
        default=False, metadata={"help": "Train an adapter instead of the full model."}
    )
    do_test: bool = field(
        default=False,
        metadata={"help": "Whether to comptue evaluation metrics on the test sets."},
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to save predictions on test sets."},
    )
    eval_output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The output directory where the evaluation of the model and checkpoints during "
            "evaluation will be written. Would use the original output_dir if not specified."
        },
    )
    optimize_from_scratch: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set, this does not load the optimizers from"
            "the given model path."
        },
    )
    optimize_from_scratch_with_loading_model: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set, it loads the model still but optimize from scratch."
        },
    )
    print_num_parameters: Optional[str] = field(
        default=False,
        metadata={"help": "If specified, prints the total number of parameters."},
    )
    compute_memory: Optional[bool] = field(
        default=False, metadata={"help": "If specified, measures the memory needed."}
    )
    compute_time: Optional[bool] = field(
        default=False, metadata={"help": "If specified, measures the time needed."}
    )


@dataclass
class ModelArguments:
    """
    Contains the arguments defining model, tokenizer, and config which we use for finetuning.
    Also, it defines which parameters of the model needs to be freezed during finetuning.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    not_load_wav2vec_checkpoint: bool = field(
        default=False, metadata={"help": "whether to load the checkpoint."}
    )
    
    base_model_name: Optional[str] = field(
        default="jonatasgrosman/wav2vec2-large-xlsr-53-german",
        metadata={
            "help": "Pretrained base model name."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    unfreeze_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to unfreeze the wav2vec encoder for training."},
    )
    freeze_embeds: bool = field(
        default=False,
        metadata={
            "help": "Whether  to freeze the conv feature extractor for wav2vec2."
        },
    )
    freeze_model: bool = field(
        default=False, metadata={"help": "Whether  to freeze the model."}
    )
    unfreeze_classifier_head: bool = field(
        default=False, metadata={"help": "Whether  to unfreeze the classifier head."}
    )
    freeze_model_but_task_embeddings: bool = field(
        default=False, metadata={"help": "freezes the whole model but task-embedding."}
    )
    unfreeze_layer_norms: bool = field(
        default=False, metadata={"help": "unfreezes the layer norms."}
    )
    unfreeze_model: bool = field(
        default=False, metadata={"help": "Whether  to unfreeze the model."}
    )
    num_labels: int = field(
        default=1, metadata={"help": "Number of outpus in classifier/regression head."}
    )
    problem_type: str = field(
        default="regression",
        metadata={"help": "Whether to perform regression or classification."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments related to data used for training and evaluation.
    """

    data_base: str = field(
        metadata={"help": "Base directory containing audio files."},
    )
    label_base: str = field(
        metadata={
            "help": "Base directory containing label files train.csv, dev.csv and test.csv."
        }
    )
    target: str = field(
        default="selfRatingDepression", metadata={"help": "Column containing labels."}
    )
    subjects: Optional[List[str]] = field(
        default="auto",
        metadata={
            "help": "List of training subjects."
        },
    )
    eval_subjects: Optional[List[str]] = field(
        default="auto",
        metadata={
            "help": "List subjects for evaluation."
        },
    )
    test_subjects: Optional[List[str]] = field(
        default="auto",
        metadata={
            "help": "List of subjects used for testing."
        },
    )
    subject_mapping: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Defines a dictionary mapping (eval and test) subjects to other (train) subjects."
        },
    )
    subject_column: str = field(
        default="subject",
        metadata={"help": "Column containing subject keys for personalisation."},
    )
    metadata_file: str = field(
        default="metadata.csv",
        metadata={"help": "File containing metadata to use for subject embeddings, must contain the subject_column."},
    )
    metadata_columns: Optional[List[str]] = field(
        default="Geschlecht",
        metadata={"help": "List of metadata columns to use as embeddings."},
    )


@dataclass
class AdapterTrainingArguments:
    """Defines the adapters parameters."""

    adapter_config_name: Optional[str] = field(
        default="meta-adapter",
        metadata={
            "help": "config name for the adapter layers, should be selected "
            f"in {sorted(ADAPTER_CONFIG_MAPPING.keys())}."
        },
    )
    task_embedding_dim: Optional[int] = field(
        default=None, metadata={"help": "task embedding dimensions."}
    )
    add_layer_norm_before_adapter: Optional[bool] = field(
        default=False, metadata={"help": "whether to have layer-norm before adapter."}
    )
    add_layer_norm_after_adapter: Optional[bool] = field(
        default=True, metadata={"help": "whether to have layer-norm after adapter."}
    )
    hidden_dim: Optional[int] = field(
        default=128,
        metadata={
            "help": "defines the default hidden dimension for " "adapter layers."
        },
    )
    reduction_factor: Optional[int] = field(
        default=16,
        metadata={
            "help": "defines the default reduction factor for " "adapter layers."
        },
    )
    non_linearity: Optional[str] = field(
        default="swish", metadata={"help": "Defines nonlinearity for adapter layers."}
    )
    projected_task_embedding_dim: Optional[int] = field(
        default=64,
        metadata={
            "help": "Defines the task embedding dimension" " after projection layer. "
        },
    )
    task_hidden_dim: Optional[int] = field(
        default=128,
        metadata={"help": "defines the hidden dimension for task embedding projector."},
    )
    conditional_layer_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Implements conditional layer norms "
            "modulated based on task embeddings."
        },
    )
    train_adapters_blocks: bool = field(
        default=True, metadata={"help": "If set, uses adapter blocks."}
    )
    unique_hyper_net: bool = field(
        default=False,
        metadata={
            "help": "If set, uses one hyper network"
            "to generates the adapter weights"
            "for all the layers."
        },
    )
    efficient_unique_hyper_net: bool = field(
        default=False,
        metadata={
            "help": "If set, uses one hyper network" "for all adapters in each layer."
        },
    )
    unique_hyper_net_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "If set, applies a layer"
            "norm after computing the "
            "embeddings for the unique "
            "hyper-net."
        },
    )
