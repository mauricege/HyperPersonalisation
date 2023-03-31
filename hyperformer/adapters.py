"""
Contains all relevant parts of https://github.com/rabeehk/hyperformer - originally spread out across multiple modules.
As we only make use of some parts, we collect them here. The parts are unchanged. 
"""
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation


@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""

    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    conditional_layer_norm = False
    hidden_dim = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    train_adapters_blocks = True


class MetaAdapterConfig(AdapterConfig):
    """Implements Meta adapter in which a hyper-network generates the parameters of
    adapter layers. In this case we have a task embeddings which is feed to the
    hyper-network to allow it generate the weights for the adapter layers."""

    task_embedding_dim = 512
    metadata_dim = None
    hidden_dim = 128
    train_task_embeddings = False
    projected_task_embedding_dim = 64
    task_hidden_dim = 128
    parametric_task_embedding = False
    # If Specified, uses one hypernet to generates the adapters weights.
    unique_hyper_net = False
    unique_hyper_net_layer_norm = True
    # We consider only one hyper-net for all the blocks of transformer.
    efficient_unique_hyper_net = False
    task_mapping = None


ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig), ("meta-adapter", MetaAdapterConfig)]
)


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}".format(
                config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())
            )
        )


class MetaLayersAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config):
        super().__init__()
        self.activation_type = config.non_linearity.lower()
        self.input_dim = config.input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter

    def apply_layer_norm(self, inputs, layer_norm_weights):
        """Applies layer norm to the inputs."""
        return torch.nn.functional.layer_norm(
            inputs,
            (self.input_dim,),
            weight=layer_norm_weights.weight,
            bias=layer_norm_weights.bias,
        )

    def call_adapter(self, inputs, adapter_weights):
        """Computes the output of the adapter layers."""
        down = F.linear(
            inputs, weight=adapter_weights.down.weight, bias=adapter_weights.down.bias
        )
        middle = get_activation(self.activation_type)(down)
        output = F.linear(
            middle, weight=adapter_weights.up.weight, bias=adapter_weights.up.bias
        )
        return output

    def forward(self, inputs, adapter_weights):
        z = (
            self.apply_layer_norm(inputs, adapter_weights.pre_norm)
            if self.add_layer_norm_before_adapter
            else inputs
        )
        outputs = self.call_adapter(z, adapter_weights)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm)
        outputs = outputs + inputs
        return outputs


class AdapterLayersHyperNet(nn.Module):
    """This module generates the weights for all the meta adapter layers
    given the task embeddings and layer id."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterLayersHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = nn.Sequential(
            linear_layer(
                config.projected_task_embedding_dim, self.input_dim * self.output_dim
            )
        )
        self.bias_generator = nn.Sequential(
            linear_layer(config.projected_task_embedding_dim, self.input_dim)
        )

    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return SamplerOutput(weight=weight, bias=bias)


class AdapterLayersOneHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, num_layers=6):
        super(AdapterLayersOneHyperNetController, self).__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.device = config.device
        self.task_embedding_dim = config.task_embedding_dim
        self.layer_id_embeddings = nn.Embedding(
            self.num_layers, self.task_embedding_dim
        ).to(self.device)
        # This is 2 types of adapters for feed-forward, and self-attention.
        self.adapters_block_type = nn.Embedding(2, self.task_embedding_dim).to(
            self.device
        )

        config.task_embedding_dim = (self.task_embedding_dim * 2) + config.metadata_dim
        self.task_hypernet = TaskHyperNet(config)
        config.task_embedding_dim = self.task_embedding_dim
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(
                config.projected_task_embedding_dim, eps=self.layer_norm_epsilon
            )
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor

        # Defines the adapters hyper-nets.
        self.up_sampler_hyper_net = AdapterLayersHyperNet(
            config, self.input_dim, self.down_sample_size
        )
        self.down_sampler_hyper_net = AdapterLayersHyperNet(
            config, self.down_sample_size, self.input_dim
        )

        # Defines the layer norms' hyper net.
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.pre_layernorm_hypernet = LayerNormHyperNet(config)
        if self.add_layer_norm_after_adapter:
            self.post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding, layer_id, block_type):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=self.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        type_id_tensor = torch.tensor(
            [block_type], dtype=torch.long, device=self.device
        )
        type_embedding = self.adapters_block_type(type_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        type_embedding = type_embedding.view(-1)
        embeddings = torch.cat(
            [
                task_embedding.view(1, -1),
                layer_embedding.view(1, -1),
                type_embedding.view(1, -1),
            ],
            axis=1,
        )
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding, layer_id):
        feed_forward_embeddings = self.get_embedding(task_embedding, layer_id, 0)
        self_attention_embeddings = self.get_embedding(task_embedding, layer_id, 1)

        # Generates the adapters weights in feed-forward.
        feed_forward_down = self.down_sampler_hyper_net(feed_forward_embeddings)
        feed_forward_up = self.up_sampler_hyper_net(feed_forward_embeddings)

        # Generates the adapter weights in self-attention.
        self_attention_down = self.down_sampler_hyper_net(self_attention_embeddings)
        self_attention_up = self.up_sampler_hyper_net(self_attention_embeddings)

        feed_forward_output = AdapterOutput(up=feed_forward_up, down=feed_forward_down)
        self_attention_output = AdapterOutput(
            up=self_attention_up, down=self_attention_down
        )

        # Generates the weights and baises for pre and post layer norms.
        if self.add_layer_norm_before_adapter:
            weight, bias = self.pre_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.pre_layernorm_hypernet(self_attention_embeddings)
            self_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)

        if self.add_layer_norm_after_adapter:
            weight, bias = self.post_layernorm_hypernet(feed_forward_embeddings)
            feed_forward_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.post_layernorm_hypernet(self_attention_embeddings)
            self_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)

        return AdapterBlockOutput(
            feed_forward=feed_forward_output, self_attention=self_attention_output
        )


@dataclass
class SamplerOutput:
    """Base class for the base and weights of each adapter."""

    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class LayerNormOutput:
    """Base class for the base and weights of the conditional
    layer norms."""

    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class AdapterOutput:
    """Base class for each adapter weights"""

    up: SamplerOutput = None
    down: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class AdapterBlockOutput:
    """
    Base class for adapter layer's outputs.
    """

    feed_forward: AdapterOutput = None
    self_attention: AdapterOutput = None


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial feeded task embeddings."""

    def __init__(self, config):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = config.task_hidden_dim
        self.projected_task_embedding_dim = config.projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(
            linear_layer(config.task_embedding_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim),
        )

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embeding_generator(task_embedding).view(-1)


class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config):
        super(LayerNormHyperNet, self).__init__()
        self.task_embedding_dim = (
            config.projected_task_embedding_dim
            if config.train_task_embeddings
            else config.task_embedding_dim
        )
        self.weight_generator = linear_layer(self.task_embedding_dim, config.input_dim)
        self.bias_generator = linear_layer(self.task_embedding_dim, config.input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)
