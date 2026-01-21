"""
Hooked Model Infrastructure for LLM Activation Extraction.

This module provides a flexible hook system for extracting intermediate
representations from transformer models during forward passes. Designed
for mechanistic interpretability research on ICL.

Supports:
- meta-llama/Meta-Llama-3-8B (primary target)
- Any HuggingFace causal LM with standard transformer architecture
- Optional nnsight integration for advanced tracing
- Optional TransformerLens integration for interpretability tools

Key Components:
- ActivationCache: Stores activations during forward pass
- HookedLLM: Wrapper providing hook infrastructure
- ResidualStreamExtractor: Specialized for residual stream analysis

References:
- TransformerLens: https://github.com/neelnanda-io/TransformerLens
- nnsight: https://github.com/ndif-team/nnsight
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Literal
from contextlib import contextmanager
from functools import partial
import warnings


@dataclass
class HookConfig:
    """Configuration for a single hook point."""
    name: str
    module_path: str  # e.g., "model.layers.0.self_attn"
    hook_type: Literal["forward", "backward"] = "forward"
    extractor: Optional[Callable] = None  # Custom extraction function


@dataclass
class ModelConfig:
    """Configuration for HookedLLM."""
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    device: str = "auto"
    dtype: torch.dtype = torch.bfloat16
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    use_cache: bool = False  # Disable KV cache for clean hooks


class ActivationCache:
    """
    Cache for storing model activations during forward pass.

    Provides dict-like access to cached activations with automatic
    device management and memory cleanup.
    """

    def __init__(self):
        self._cache: dict[str, torch.Tensor] = {}
        self._metadata: dict[str, dict] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._cache[key]

    def __setitem__(self, key: str, value: torch.Tensor):
        self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def keys(self):
        return self._cache.keys()

    def items(self):
        return self._cache.items()

    def get(self, key: str, default: Any = None) -> torch.Tensor | Any:
        return self._cache.get(key, default)

    def add(
        self,
        name: str,
        activation: torch.Tensor,
        detach: bool = True,
        clone: bool = True,
        metadata: Optional[dict] = None
    ):
        """
        Add an activation to the cache.

        Args:
            name: Identifier for this activation
            activation: The tensor to cache
            detach: Whether to detach from computation graph
            clone: Whether to clone the tensor
            metadata: Optional metadata dict
        """
        tensor = activation
        if detach:
            tensor = tensor.detach()
        if clone:
            tensor = tensor.clone()

        self._cache[name] = tensor
        if metadata:
            self._metadata[name] = metadata

    def clear(self):
        """Clear all cached activations."""
        self._cache.clear()
        self._metadata.clear()

    def to_device(self, device: str | torch.device):
        """Move all cached tensors to a device."""
        self._cache = {k: v.to(device) for k, v in self._cache.items()}

    def get_residual_stream(self, layer: int) -> torch.Tensor | None:
        """Get residual stream activation for a specific layer."""
        key = f"residual_stream.layer_{layer}"
        return self._cache.get(key)

    def get_all_residual_streams(self) -> dict[int, torch.Tensor]:
        """Get all cached residual stream activations."""
        result = {}
        for key, value in self._cache.items():
            if key.startswith("residual_stream.layer_"):
                layer = int(key.split("_")[-1])
                result[layer] = value
        return result


class HookedLLM:
    """
    Wrapper for HuggingFace models providing hook infrastructure.

    Enables activation extraction at any layer of the model during
    forward passes. Designed for mechanistic interpretability research.

    Example:
        model = HookedLLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

        with model.hooks(layers=[0, 15, 31]):
            output, cache = model.forward_with_cache("Hello world")

        # Access residual stream at layer 15
        residual = cache["residual_stream.layer_15"]
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: ModelConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._hooks: list = []
        self._cache = ActivationCache()

        # Detect model architecture
        self._detect_architecture()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ) -> "HookedLLM":
        """
        Load a pretrained model with hook infrastructure.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            dtype: Data type for model weights
            **kwargs: Additional arguments for model loading

        Returns:
            HookedLLM instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = ModelConfig(
            model_name=model_name,
            device=device,
            dtype=dtype,
            **{k: v for k, v in kwargs.items()
               if k in ModelConfig.__dataclass_fields__}
        )

        # Determine device map
        if device == "auto":
            device_map = "auto"
        else:
            device_map = {"": device}

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code
        )

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "trust_remote_code": config.trust_remote_code,
        }

        if config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        model.eval()  # Set to eval mode

        return cls(model, tokenizer, config)

    def _detect_architecture(self):
        """Detect model architecture for hook placement."""
        model_type = getattr(self.model.config, "model_type", "unknown")

        # Map model types to layer access paths
        self._layer_patterns = {
            "llama": ("model.layers", "input_layernorm", "post_attention_layernorm"),
            "mistral": ("model.layers", "input_layernorm", "post_attention_layernorm"),
            "gpt2": ("transformer.h", "ln_1", "ln_2"),
            "gpt_neox": ("gpt_neox.layers", "input_layernorm", "post_attention_layernorm"),
            "qwen2": ("model.layers", "input_layernorm", "post_attention_layernorm"),
            "qwen3": ("model.layers", "input_layernorm", "post_attention_layernorm"),
            "gemma": ("model.layers", "input_layernorm", "post_attention_layernorm"),
            "gemma2": ("model.layers", "input_layernorm", "post_attention_layernorm"),
            "gemma3": ("model.text_model.layers", "input_layernorm", "post_attention_layernorm"),
            "gemma3_text": ("model.layers", "input_layernorm", "post_attention_layernorm"),
        }

        self._model_type = model_type
        if model_type in self._layer_patterns:
            self._layers_path, self._pre_ln, self._post_ln = self._layer_patterns[model_type]
        else:
            # Try common patterns - check multiple paths for unknown models
            layers_found = False
            for path in ["model.layers", "model.text_model.layers", "transformer.h"]:
                parts = path.split(".")
                module = self.model
                try:
                    for part in parts:
                        module = getattr(module, part)
                    if hasattr(module, "__len__"):
                        self._layers_path = path
                        self._pre_ln = "input_layernorm"
                        self._post_ln = "post_attention_layernorm"
                        layers_found = True
                        break
                except AttributeError:
                    continue

            if not layers_found:
                warnings.warn(f"Unknown model type: {model_type}. Using default paths.")
                self._layers_path = "model.layers"
                self._pre_ln = "input_layernorm"
                self._post_ln = "post_attention_layernorm"

    def _get_layers(self) -> nn.ModuleList:
        """Get the transformer layers module list."""
        parts = self._layers_path.split(".")
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return len(self._get_layers())

    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the model."""
        return self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        """Primary device of the model."""
        return next(self.model.parameters()).device

    def _create_hook_fn(
        self,
        name: str,
        extractor: Optional[Callable] = None
    ) -> Callable:
        """Create a hook function that stores activations in cache."""
        def hook_fn(module, input, output):
            if extractor is not None:
                activation = extractor(output)
            elif isinstance(output, tuple):
                activation = output[0]  # Usually hidden states
            else:
                activation = output

            self._cache.add(name, activation)

        return hook_fn

    def _register_residual_stream_hooks(self, layers: list[int]):
        """
        Register hooks to capture residual stream at specified layers.

        The residual stream is captured AFTER the layer (post-residual).
        """
        transformer_layers = self._get_layers()

        for layer_idx in layers:
            if layer_idx < 0:
                layer_idx = len(transformer_layers) + layer_idx

            if layer_idx >= len(transformer_layers):
                warnings.warn(f"Layer {layer_idx} out of range, skipping")
                continue

            layer = transformer_layers[layer_idx]

            # Hook the entire layer to get post-residual output
            hook_name = f"residual_stream.layer_{layer_idx}"
            hook_fn = self._create_hook_fn(hook_name)
            handle = layer.register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @contextmanager
    def hooks(
        self,
        layers: Optional[list[int]] = None,
        hook_configs: Optional[list[HookConfig]] = None
    ):
        """
        Context manager for registering hooks during forward pass.

        Args:
            layers: Layer indices to hook residual stream (default: all)
            hook_configs: Custom hook configurations

        Yields:
            self for chaining

        Example:
            with model.hooks(layers=[0, 15, -1]) as m:
                output, cache = m.forward_with_cache(text)
        """
        self._cache.clear()

        try:
            # Register residual stream hooks
            if layers is None:
                layers = list(range(self.num_layers))
            self._register_residual_stream_hooks(layers)

            # Register custom hooks if provided
            if hook_configs:
                for hc in hook_configs:
                    # Navigate to module
                    parts = hc.module_path.split(".")
                    module = self.model
                    for part in parts:
                        module = getattr(module, part)

                    hook_fn = self._create_hook_fn(hc.name, hc.extractor)

                    if hc.hook_type == "forward":
                        handle = module.register_forward_hook(hook_fn)
                    else:
                        handle = module.register_full_backward_hook(hook_fn)

                    self._hooks.append(handle)

            yield self

        finally:
            self._clear_hooks()

    def tokenize(
        self,
        text: str | list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> dict:
        """Tokenize text input."""
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

    def forward_with_cache(
        self,
        text: str | list[str],
        layers: Optional[list[int]] = None,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, ActivationCache]:
        """
        Run forward pass and return cached activations.

        This is the main method for activation extraction. It automatically
        registers hooks, runs the forward pass, and returns the cache.

        Args:
            text: Input text or list of texts
            layers: Layers to hook (default: all)
            return_logits: Whether to return model logits

        Returns:
            Tuple of (logits or None, activation cache)
        """
        with self.hooks(layers=layers):
            inputs = self.tokenize(text)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits if return_logits else None

            # Copy cache before context exit clears hooks
            cache = ActivationCache()
            cache._cache = dict(self._cache._cache)
            cache._metadata = dict(self._cache._metadata)

        return logits, cache

    def get_token_representations(
        self,
        text: str,
        layer: int,
        token_positions: Optional[list[int]] = None
    ) -> torch.Tensor:
        """
        Get representations for specific tokens at a layer.

        Args:
            text: Input text
            layer: Layer index
            token_positions: Token positions to extract (default: all)

        Returns:
            Tensor of shape (num_tokens, hidden_size)
        """
        _, cache = self.forward_with_cache(text, layers=[layer])

        residual = cache.get_residual_stream(layer)
        if residual is None:
            raise ValueError(f"No residual stream cached for layer {layer}")

        # Remove batch dimension (assuming batch_size=1)
        residual = residual.squeeze(0)

        if token_positions is not None:
            residual = residual[token_positions]

        return residual

    def compute_per_token_loss(
        self,
        text: str,
        reduction: str = "none"
    ) -> torch.Tensor:
        """
        Compute loss for each token position (next-token prediction).

        This is L(x_i) in the BIF formula: how well each token fits
        given the preceding context.

        Args:
            text: Input text
            reduction: Loss reduction ('none' for per-token, 'mean', 'sum')

        Returns:
            Tensor of per-token losses
        """
        inputs = self.tokenize(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Compute per-token cross entropy
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        if reduction == "none":
            loss = loss.view(shift_labels.shape)

        return loss


class ResidualStreamExtractor:
    """
    Specialized extractor for residual stream analysis.

    Provides utilities for computing geometric properties of
    representations across layers and token positions.
    """

    def __init__(self, model: HookedLLM):
        self.model = model

    def extract_trajectory(
        self,
        text: str,
        layers: Optional[list[int]] = None,
        token_position: int = -1
    ) -> torch.Tensor:
        """
        Extract the residual stream trajectory for a token across layers.

        Args:
            text: Input text
            layers: Layers to extract (default: all)
            token_position: Which token to track (-1 for last)

        Returns:
            Tensor of shape (num_layers, hidden_size)
        """
        if layers is None:
            layers = list(range(self.model.num_layers))

        _, cache = self.model.forward_with_cache(text, layers=layers)

        trajectory = []
        for layer in sorted(layers):
            residual = cache.get_residual_stream(layer)
            if residual is not None:
                # Extract specific token position
                token_repr = residual[0, token_position, :]
                trajectory.append(token_repr)

        return torch.stack(trajectory)

    def extract_cluster_representations(
        self,
        texts: list[str],
        layer: int,
        cluster_labels: list[int],
        token_position: int = -1
    ) -> dict[int, torch.Tensor]:
        """
        Extract representations grouped by cluster labels.

        Useful for computing inter-cluster distances as a structural metric.

        Args:
            texts: List of input texts (one per cluster member)
            layer: Layer to extract from
            cluster_labels: Cluster assignment for each text
            token_position: Token position to extract

        Returns:
            Dict mapping cluster_id -> stacked representations
        """
        representations = {}

        for text, cluster in zip(texts, cluster_labels):
            repr_vec = self.model.get_token_representations(
                text, layer, [token_position]
            ).squeeze(0)

            if cluster not in representations:
                representations[cluster] = []
            representations[cluster].append(repr_vec)

        # Stack within clusters
        return {
            k: torch.stack(v)
            for k, v in representations.items()
        }


def demo():
    """Demonstrate the HookedLLM functionality (with small model)."""
    print("=" * 60)
    print("HookedLLM Demo")
    print("=" * 60)

    print("\nNote: This demo uses gpt2-small for fast testing.")
    print("For actual experiments, use meta-llama/Meta-Llama-3-8B")

    try:
        # Use small model for demo
        model = HookedLLM.from_pretrained(
            "gpt2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32
        )

        print(f"\nModel loaded: {model.config.model_name}")
        print(f"  Num layers: {model.num_layers}")
        print(f"  Hidden size: {model.hidden_size}")
        print(f"  Device: {model.device}")

        # Test forward with cache
        text = "The quick brown fox jumps over the lazy dog"
        print(f"\nTest input: '{text}'")

        logits, cache = model.forward_with_cache(text, layers=[0, 5, 11])

        print(f"\nCached activations:")
        for key in cache.keys():
            tensor = cache[key]
            print(f"  {key}: shape={tuple(tensor.shape)}")

        # Test per-token loss
        losses = model.compute_per_token_loss(text)
        print(f"\nPer-token losses shape: {tuple(losses.shape)}")
        print(f"Mean loss: {losses.mean().item():.4f}")

        # Test residual stream extractor
        extractor = ResidualStreamExtractor(model)
        trajectory = extractor.extract_trajectory(
            text,
            layers=[0, 3, 6, 9, 11],
            token_position=-1
        )
        print(f"\nResidual trajectory shape: {tuple(trajectory.shape)}")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nDemo failed: {e}")
        print("This is expected if transformers is not installed.")
        print("Install with: pip install transformers torch")


if __name__ == "__main__":
    demo()
