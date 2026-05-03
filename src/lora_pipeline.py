"""
lora_pipeline.py — LoRA / PEFT Fine-Tuned Model Pipeline
==========================================================

Responsibilities:
  1. Load a base causal-LM (e.g., Mistral-7B-Instruct).
  2. Apply a PEFT/LoRA adapter on top of it.
  3. Expose a `generate()` method for inference.

Design Trade-offs:
  • We use a *placeholder* PEFT config here.  In production you would
    point `peft_model_path` to a directory containing your trained
    adapter weights (adapter_config.json + adapter_model.bin).
  • BitsAndBytes 4-bit quantization is included to make the 7B model
    runnable on consumer GPUs (≈ 6 GB VRAM).  Remove the
    `BitsAndBytesConfig` block for full-precision inference on larger
    hardware.
  • `torch.inference_mode()` is used instead of `torch.no_grad()` for
    a marginal speed improvement during generation.

Fine-Tuning Workflow (out of scope for this scaffold):
  1. Prepare a domain-specific instruction dataset (JSONL).
  2. Use `peft.get_peft_model(base_model, lora_config)` to wrap the
     base model with trainable LoRA matrices.
  3. Train with HuggingFace `Trainer` or `SFTTrainer`.
  4. Save only the adapter weights:  `model.save_pretrained(path)`.
  5. At inference time, load the adapter with
     `PeftModel.from_pretrained(base_model, path)`.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Configuration defaults                                             #
# ------------------------------------------------------------------ #

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


class LoRAPipeline:
    """Manages a PEFT/LoRA-augmented causal language model.

    Parameters
    ----------
    base_model_name : str
        HuggingFace Hub ID for the base model.
    peft_model_path : str | None
        Path to trained LoRA adapter weights.  If ``None``, a fresh
        (untrained) LoRA config is applied — useful for scaffolding
        and smoke-testing the pipeline before training.
    load_in_4bit : bool
        Whether to quantize the base model to 4-bit via bitsandbytes.
    device : str
        Target device (``"auto"``, ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        peft_model_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "auto",
    ) -> None:
        self.base_model_name = base_model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._peft_model_path = peft_model_path
        self._load_in_4bit = load_in_4bit
        self._is_loaded = False

    # -------------------------------------------------------------- #
    #  Lazy Loading                                                    #
    # -------------------------------------------------------------- #

    def load(self) -> None:
        """Load the base model + LoRA adapter into memory.

        This is intentionally separated from ``__init__`` so the
        FastAPI app can start quickly and defer the heavy model load
        to the first request (or to a startup event).
        """
        if self._is_loaded:
            logger.info("Model already loaded — skipping.")
            return

        logger.info("Loading tokenizer for %s …", self.base_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        # Ensure the tokenizer has a pad token (Mistral doesn't by default).
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # ----------------------------------------------------------
        # Quantization config (optional 4-bit via bitsandbytes).
        # Dramatically reduces VRAM usage: 7B model fits in ~6 GB.
        # ----------------------------------------------------------
        quantization_config = None
        if self._load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info("Loading base model: %s …", self.base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # ----------------------------------------------------------
        # Apply LoRA adapter
        # ----------------------------------------------------------
        if self._peft_model_path:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PRODUCTION PATH — load trained adapter weights.
            #
            # The `peft_model_path` directory should contain:
            #   • adapter_config.json   (LoRA hyper-parameters)
            #   • adapter_model.bin     (trained LoRA weight deltas)
            #
            # These are produced by `model.save_pretrained(path)`
            # after fine-tuning with HuggingFace PEFT.
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.info(
                "Loading trained LoRA adapter from %s …", self._peft_model_path
            )
            self._model = PeftModel.from_pretrained(
                base_model,
                self._peft_model_path,
            )
        else:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PLACEHOLDER / SCAFFOLD PATH — apply a fresh LoRA config
            # with *random* adapter weights.  The model will behave
            # like the base model because untrained LoRA deltas are
            # near-zero.  Replace this with a real adapter path once
            # you have trained one.
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            logger.warning(
                "No trained adapter path provided — using a PLACEHOLDER "
                "LoRA config.  Responses will match the base model."
            )
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,                    # rank of the low-rank matrices
                lora_alpha=16,          # scaling factor
                lora_dropout=0.05,      # dropout on LoRA layers
                target_modules=[        # Mistral attention projections
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ],
                bias="none",
            )
            self._model = get_peft_model(base_model, lora_config)

        self._model.eval()
        self._is_loaded = True
        logger.info("LoRA pipeline ready.")

    # -------------------------------------------------------------- #
    #  Generation                                                      #
    # -------------------------------------------------------------- #

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """Generate a completion for the given *prompt*.

        Returns only the newly generated tokens (the prompt is stripped).
        """
        if not self._is_loaded:
            self.load()

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        # Generate
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                generation_config=gen_config,
            )

        # Decode only the *new* tokens (skip the input prompt).
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    # -------------------------------------------------------------- #
    #  Utilities                                                       #
    # -------------------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_trainable_parameters(self) -> dict:
        """Return a summary of trainable vs. total parameters."""
        if not self._is_loaded:
            return {"error": "Model not loaded yet."}

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": f"{100 * trainable / total:.4f}%",
        }
