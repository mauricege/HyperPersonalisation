import sys
import torch
import pyreadstat
import datasets
import pandas as pd
import json
import logging
import os
from glob import glob
from os.path import join, basename
from os import listdir
from pathlib import Path

from transformers import AutoTokenizer, HfArgumentParser, set_seed, Wav2Vec2Processor
from transformers.trainer_utils import EvaluationStrategy

from hyperpersonalisation.third_party.models import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Config,
)
from hyperpersonalisation.third_party.trainers import Wav2Vec2Trainer
from hyperformer.adapters import MetaAdapterConfig
from hyperpersonalisation.data import AutoTask, DataCollatorCTCWithPadding
from hyperpersonalisation.third_party.utils import (
    check_output_dir,
)
from hyperpersonalisation.metrics import build_compute_metrics_fn
from hyperpersonalisation.training_args import (
    Seq2SeqTrainingArguments,
    ModelArguments,
    DataTrainingArguments,
    AdapterTrainingArguments,
)
from hyperpersonalisation.utils import (
    freezing_params,
    last_checkpoint,
    best_checkpoint,
    save_metrics,
    get_training_args,
)


logger = logging.getLogger(__name__)


def main(fold=0):
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            AdapterTrainingArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        logger.warning("Loading from config: %s", sys.argv[1])
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            adapter_args,
        ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    training_args.output_dir = join(training_args.output_dir, str(fold))
    check_output_dir(training_args)

    if os.path.exists(model_args.model_name_or_path):
        model_args.model_name_or_path = best_checkpoint(os.path.join(model_args.model_name_or_path, str(fold)))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    logger.info(f"Load metadata from {data_args.metadata_file}.")
    metadata = pd.read_csv(data_args.metadata_file)
    metadata = metadata.fillna(metadata.mean())

    metadata = metadata[[data_args.subject_column] + data_args.metadata_columns]
    logger.info(f"Dropping non-numeric columns from metadata.")
    metadata = pd.concat(
        (
            metadata[data_args.subject_column],
            metadata[data_args.metadata_columns].select_dtypes(["number"]),
        ),
        axis=1,
    )
    data_args.metadata_columns = list(metadata.columns[1:])
    logger.info(f"New metadata columns {data_args.metadata_columns}")

    if data_args.subjects == "auto":
        data_args.subject_embeddings = {}
        labels = pd.read_csv(join(data_args.label_base, str(fold), "train.csv"))
        labels = labels["subject"].drop_duplicates()
        metadata_train = pd.merge(labels, metadata, on="subject")
        data_args.subjects = list(metadata_train.subject.values)

    if data_args.eval_subjects == "auto":
        labels = pd.read_csv(join(data_args.label_base, str(fold), "dev.csv"))
        labels = labels["subject"].drop_duplicates()
        metadata_eval = pd.merge(labels, metadata, on="subject")
        data_args.eval_subjects = list(metadata_eval.subject.values)

    if data_args.test_subjects == "auto":
        labels = pd.read_csv(join(data_args.label_base, str(fold), "test.csv"))
        labels = labels["subject"].drop_duplicates()
        metadata_test = pd.merge(labels, metadata, on="subject")
        data_args.test_subjects = list(metadata_test.subject.values)

    df = pd.concat(
        map(pd.read_csv, glob(f"{join(data_args.label_base, str(fold))}/*.csv"))
    )
    df = pd.merge(df, metadata, on="subject")

    config = Wav2Vec2Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    config.num_labels = model_args.num_labels
    config.problem_type = model_args.problem_type
    extra_model_params = ("train_adapters",)
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(
                config, p
            ), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    if training_args.train_adapters:
        adapter_config = MetaAdapterConfig
        adapter_config.input_dim = config.hidden_size
        adapter_config.metadata_dim = len(data_args.metadata_columns)
        adapter_config.tasks = (
            data_args.subjects + data_args.eval_subjects + data_args.test_subjects
        )
        adapter_config.task_to_embeddings = data_args.subject_embeddings
        extra_adapter_params = (
            "task_embedding_dim",
            "add_layer_norm_before_adapter",
            "add_layer_norm_after_adapter",
            "reduction_factor",
            "hidden_dim",
            "non_linearity",
            "projected_task_embedding_dim",
            "task_hidden_dim",
            "conditional_layer_norm",
            "train_adapters_blocks",
            "unique_hyper_net",
            "unique_hyper_net_layer_norm",
            "efficient_unique_hyper_net",
        )
        for p in extra_adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p):
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(
                    f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute"
                )
        adapter_config.device = training_args.device
    else:
        adapter_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_name
        if model_args.base_model_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if model_args.not_load_wav2vec_checkpoint:
        model = Wav2Vec2ForSequenceClassification(
            config=config, adapter_config=adapter_config
        )
    else:
        logger.warning("model path loaded from : %s", model_args.model_name_or_path)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            adapter_config=adapter_config,
        )
    processor = Wav2Vec2Processor.from_pretrained(model_args.base_model_name)
    target_sampling_rate = processor.feature_extractor.sampling_rate

    if training_args.do_train:
        freezing_params(model, training_args, model_args, adapter_args)

    if training_args.print_num_parameters:
        logger.info(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("Parameter name %s", name)
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total parameters %s", total_params)
    dataset_class = AutoTask
    if training_args.do_train:
        train_datasets = [
            dataset_class.get(
                "Regression",
                df=df,
                data_base=data_args.data_base,
                subject_column=data_args.subject_column,
                subject=group,
                target=data_args.target,
                processor=processor,
                metadata_columns=data_args.metadata_columns,
                target_sampling_rate=target_sampling_rate,
                seed=training_args.data_seed,
            ).get_dataset()
            for group in data_args.subjects
        ]
        dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
        train_dataset = datasets.concatenate_datasets(train_datasets)
    training_args.remove_unused_columns = False
    eval_datasets = (
        {
            group: dataset_class.get(
                "Regression",
                df=df,
                data_base=data_args.data_base,
                subject_column=data_args.subject_column,
                subject=group,
                target=data_args.target,
                processor=processor,
                metadata_columns=data_args.metadata_columns,
                target_sampling_rate=target_sampling_rate,
                seed=training_args.data_seed,
            ).get_dataset()
            for group in data_args.eval_subjects
        }
        if training_args.do_eval
        or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        {
            group: dataset_class.get(
                "Regression",
                df=df,
                data_base=data_args.data_base,
                subject_column=data_args.subject_column,
                subject=group,
                target=data_args.target,
                processor=processor,
                metadata_columns=data_args.metadata_columns,
                target_sampling_rate=target_sampling_rate,
                seed=training_args.data_seed,
            ).get_dataset()
            for group in data_args.test_subjects
        }
        if training_args.do_test
        else None
    )
    # Defines the metrics for evaluation.
    compute_metrics_fn = build_compute_metrics_fn("Regression", data_args.eval_subjects)
    # Defines the trainer.
    training_args.fp16 = False
    trainer = Wav2Vec2Trainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets,
        data_collator=DataCollatorCTCWithPadding(
            processor, padding=True, max_length=5 * target_sampling_rate
        ),
        compute_metrics=None,
        multi_task_compute_metrics=compute_metrics_fn,
        data_args=data_args,
        dataset_sizes=dataset_sizes if training_args.do_train else None,
        adapter_config=adapter_config,
    )
    arguments = get_training_args([model_args, data_args, training_args, adapter_args])
    save_metrics("arguments", arguments, training_args.output_dir)

    # Trains the model.
    if training_args.do_train:
        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        trainer.train(
            model_path=last_checkpoint(training_args.output_dir)
            if (
                os.path.exists(training_args.output_dir)
                and not training_args.optimize_from_scratch
            )
            else None,
        )
        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for all_reduce to complete
            end.record()
            total_time = {"total_time": start.elapsed_time(end)}
            print("###### total_time ", total_time)
        trainer.save_model()

        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    all_metrics = {}
    if training_args.do_eval or training_args.do_test:
        last_checkpoint_path = best_checkpoint(training_args.output_dir)
        config = Wav2Vec2Config.from_pretrained(
            last_checkpoint_path, cache_dir=model_args.cache_dir
        )
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            last_checkpoint_path,
            from_tf=".ckpt" in training_args.output_dir,
            config=config,
            cache_dir=model_args.cache_dir,
            adapter_config=adapter_config,
        )

        trainer = Wav2Vec2Trainer(
            model=model,
            config=config,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_datasets,
            data_collator=DataCollatorCTCWithPadding(
                processor, padding=True, max_length=5 * target_sampling_rate
            ),
            compute_metrics=None,
            multi_task_compute_metrics=compute_metrics_fn,
            data_args=data_args,
            dataset_sizes=dataset_sizes if training_args.do_train else None,
            adapter_config=adapter_config,
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        save_metrics("val", metrics, training_args.output_dir)
        all_metrics.update(metrics)

    if training_args.do_test and not training_args.do_predict:
        metrics = trainer.evaluate(test_dataset)
        save_metrics("test", metrics, training_args.output_dir)
        all_metrics.update(metrics)

    if training_args.do_predict:
        compute_metrics_fn = (
            build_compute_metrics_fn("Regression", data_args.test_subjects)
        )
        trainer.multi_task_compute_metrics = compute_metrics_fn
        predictions = trainer.predict(test_dataset)
        metrics = {
            eval_task + "_" + k: v
            for eval_task, output in predictions.items()
            for k, v in output.metrics.items()
        }
        filenames, preds, trues = [], [], []
        save_metrics("test", metrics, training_args.output_dir)
        for subject in predictions:
            filenames += test_dataset[subject].data["filename"]
            preds += list(predictions[subject].predictions.squeeze())
            trues += list(predictions[subject].label_ids)
        df = pd.DataFrame({"filename": filenames, "prediction": preds, "true": trues})
        df.to_csv(join(training_args.output_dir, "predictions.test.csv"))

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print("Memory utilization", peak_memory, "MB")
        memory_usage = {"peak_memory": peak_memory}
    return all_metrics




if __name__ == "__main__":
    jobid = os.getenv("SLURM_ARRAY_TASK_ID") or os.getenv("FOLD") or ""
    main(jobid)
    
