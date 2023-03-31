import glob
import os
from dataclasses import asdict
from logging import getLogger
from hyperpersonalisation.third_party.utils import (
    assert_all_frozen,
    freeze_embeds,
    freeze_params,
    save_json,
    load_json,
)
from torch.nn import LayerNorm

from hyperformer.adapters import (
    AdapterLayersOneHyperNetController,
)

logger = getLogger(__name__)

def save_metrics(split, metrics, output_dir):
    """
    Prints and logs metrics.

    Args:
        split: trian/val/test.
        metrics: metrics dict
        output_dir: where to save the metrics
    """
    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key}: {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}.json"))


def get_training_args(arguments_list):
    """
    Concatenate all training arguments except evaluation strategy which
    is not Json serializable.
    Args:
        arguments_list: list of dataclasses.
    Return:
        arguments: concatenated arguments.
    """
    all_arguments = {}
    for arguments in arguments_list:
        all_arguments.update(asdict(arguments))
    all_arguments.pop("evaluation_strategy")
    all_arguments.pop("logging_strategy")
    all_arguments.pop("save_strategy")
    all_arguments.pop("hub_strategy")
    all_arguments.pop("optim")
    all_arguments.pop("lr_scheduler_type")
    return all_arguments


def last_checkpoint(output_dir):
    """
    Find the last checkpoint in output_dir
    
    """
    paths = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
    if len(paths) == 0:
        return output_dir
    else:
        return paths[-1]

def best_checkpoint(output_dir):
    """
    Find the best checkpoint in output_dir. If no best checkpoint found, return last checkpoint.
    """
    state_path = os.path.join(last_checkpoint(output_dir), "trainer_state.json")
    if os.path.exists(state_path):
        trainer_state = load_json(state_path)
        best_chkpt = trainer_state["best_model_checkpoint"]
        if best_chkpt is not None:
            best_chkpt = os.path.normpath(best_chkpt).split(os.sep)[-1]
            best_chkpt = os.path.join(output_dir, best_chkpt)
            return best_chkpt
        else:
            return last_checkpoint(output_dir)
    else:
        return output_dir


def freezing_params(model, training_args, model_args, adapter_args):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      training_args: defines the training arguments.
      model_args: defines the model arguments.
      adapter_args: defines the adapters arguments.
    """
    
    if training_args.train_adapters:
        freeze_params(model)
       
        if adapter_args.efficient_unique_hyper_net:
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (AdapterLayersOneHyperNetController)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
    if model_args.freeze_model:
        freeze_params(model)

    if model_args.unfreeze_classifier_head:
        for param in model.projector.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Unfreezes layer norms.
    if model_args.unfreeze_layer_norms:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, LayerNorm):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
    
    if model_args.freeze_embeds:
        model.freeze_feature_encoder()

    if model_args.unfreeze_encoder:
        for param in model.wav2vec2.feature_projection.parameters():
            param.requires_grad = True
        for param in model.wav2vec2.encoder.parameters():
            param.requires_grad = True
        if model.wav2vec2.adapter is not None:
            for param in model.wav2vec2.adapter.parameters():
                param.requires_grad = True

    if model_args.unfreeze_model:
        for param in model.parameters():
            param.requires_grad = True