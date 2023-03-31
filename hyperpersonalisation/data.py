import abc
import functools
import logging
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from os.path import join
from typing import Callable, Dict, List, Mapping, Union, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2Processor

from hyperpersonalisation import metrics

logger = logging.getLogger(__name__)


class AbstractSpeechPersonalisationDataset(abc.ABC):

    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    target: str = NotImplemented

    def __init__(
        self,
        df,
        target,
        subject,
        processor,
        metadata_columns,
        target_sampling_rate=16000,
        seed=42,
    ):
        self.seed = seed
        self.subject = subject
        self.df = df[df.subject == subject]
        self.target_sampling_rate = target_sampling_rate
        self.processor = processor
        self.metadata_columns = metadata_columns
       
    def get_dataset(self):

        dataset = Dataset.from_pandas(self.df)
        return dataset.map(
            self.preprocessor,
            remove_columns=dataset.column_names,
        )

    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(
            sampling_rate, self.target_sampling_rate
        )
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def preprocessor(self, example):
        speech_list = self.speech_file_to_array_fn(example["filename"])
        target_list = float(
            example[self.target]
        )  # Do any preprocessing on your float/integer data
        metadata = np.array([float(example[c]) for c in self.metadata_columns])

        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
        result["labels"] = target_list
        result["filename"] = example["filename"]
        result["subject"] = self.subject
        result["metadata"] = metadata

        return result


class RegressionSpeechPersonalisationDataset(AbstractSpeechPersonalisationDataset):
    metrics = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]

    def __init__(
        self,
        # label_base,
        df,
        data_base,
        subject_column,
        subject,
        target,
        processor,
        target_sampling_rate,
        metadata_columns,
        seed=42,
        apply_subject_fn=lambda x: x,
    ):
        self.target = target
        all_files = list(glob(f"{data_base}/**/*.wav", recursive=True))


        df["filename"] = df["filename"].apply(lambda x: join(data_base, x))
        df["subject"] = df[subject_column].apply(apply_subject_fn)
        df = df[df["filename"].isin(all_files)]

        super().__init__(
            df,
            target=self.target,
            subject=subject,
            processor=processor,
            metadata_columns=metadata_columns,
            target_sampling_rate=target_sampling_rate,
            seed=seed,
        )


TASK_MAPPING = OrderedDict(
    [
        ("Regression", RegressionSpeechPersonalisationDataset),
    ]
)


class AutoTask:
    @classmethod
    def get(
        self,
        task_name,
        df,
        data_base,
        subject_column,
        subject,
        target,
        processor,
        metadata_columns,
        target_sampling_rate,
        seed=42,
        apply_subject_fn=lambda x: x,
    ):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](
                # label_base,
                df,
                data_base,
                subject_column,
                subject,
                target,
                processor,
                target_sampling_rate,
                metadata_columns,
                seed,
                apply_subject_fn,
            )
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"][0]} for feature in features
        ]
        label_features = [feature["labels"] for feature in features]
        task_features = [feature["subject"] for feature in features]
        metadata_features = [feature["metadata"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        batch["task"] = task_features[0] # there should be only one task per batch
        batch["task_embedding"] = torch.tensor(metadata_features[0], dtype=d_type)
        

        return batch