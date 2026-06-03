"""
lora_pipeline.py — LoRA / PEFT Fine-Tuned Model Pipeline
==========================================================

Upgrade features:
  1. Dynamic adapter loading (scans models/adapters).
  2. Adapter switching on-the-fly.
  3. Quantized inference support (4-bit on GPU).
  4. Automatic CPU/GPU fallback with model scaling.
  5. Caches models and adapters to the models/ directory.
"""

from __future__ import annotations

import logging
import os
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

DEFAULT_GPU_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
DEFAULT_CPU_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

MODELS_DIR = Path("./models")
BASE_MODELS_DIR = MODELS_DIR / "base_models"
ADAPTERS_DIR = MODELS_DIR / "adapters"


class LoRAPipeline:
    """Manages a causal language model with dynamic PEFT/LoRA adapter loading and CPU/GPU fallback."""

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "auto",
    ) -> None:
        self.device_setting = device
        self.load_in_4bit = load_in_4bit
        
        # Determine actual device
        self.device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        
        # Dynamic fallback for base model if running on CPU
        if base_model_name:
            self.base_model_name = base_model_name
        else:
            self.base_model_name = DEFAULT_GPU_MODEL if self.device == "cuda" else DEFAULT_CPU_MODEL
            
        if self.device == "cpu" and ("7B" in self.base_model_name or "8B" in self.base_model_name):
            logger.warning(
                "Large model '%s' requested on CPU. Automatically falling back to light model '%s' to prevent OOM.",
                self.base_model_name,
                DEFAULT_CPU_MODEL
            )
            self.base_model_name = DEFAULT_CPU_MODEL

        self._model: Optional[PeftModel | AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._is_loaded = False
        self._active_adapter: str = "base"
        self._loaded_adapters: Dict[str, str] = {}  # name -> path

        # Initialize folders
        os.makedirs(BASE_MODELS_DIR, exist_ok=True)
        os.makedirs(ADAPTERS_DIR, exist_ok=True)

    # -------------------------------------------------------------- #
    #  Lazy Loading                                                    #
    # -------------------------------------------------------------- #

    def load(self) -> None:
        """Load tokenizer, base model, and scan for LoRA adapters."""
        if self._is_loaded:
            return

        logger.info("Initializing model: %s on device: %s", self.base_model_name, self.device)
        
        # 1. Load Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            cache_dir=str(BASE_MODELS_DIR),
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 2. Quantization config (only valid on GPU)
        quantization_config = None
        if self.device == "cuda" and self.load_in_4bit:
            logger.info("Enabling 4-bit Quantization (bitsandbytes).")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # 3. Load Base Model
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "auto" if self.device == "cuda" else "cpu"
        
        logger.info("Loading base model into memory...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=dtype,
            cache_dir=str(BASE_MODELS_DIR),
            trust_remote_code=True,
        )

        # 4. Wrap in PEFT
        # We start by checking if there are any adapters in models/adapters
        adapter_dirs = [d for d in ADAPTERS_DIR.iterdir() if d.is_dir()]
        
        first_adapter_loaded = False
        self._model = base_model

        for d in adapter_dirs:
            # Check if it has config
            if (d / "adapter_config.json").exists():
                adapter_name = d.name
                adapter_path = str(d)
                try:
                    if not first_adapter_loaded:
                        logger.info("Loading first LoRA adapter '%s' from %s", adapter_name, adapter_path)
                        self._model = PeftModel.from_pretrained(
                            base_model,
                            adapter_path,
                            adapter_name=adapter_name,
                        )
                        first_adapter_loaded = True
                    else:
                        logger.info("Loading subsequent LoRA adapter '%s' from %s", adapter_name, adapter_path)
                        self._model.load_adapter(adapter_path, adapter_name)
                    
                    self._loaded_adapters[adapter_name] = adapter_path
                except Exception as e:
                    logger.error("Failed to load adapter %s: %s", adapter_name, e)

        # If no adapters were loaded from disk, we wrap with a dummy adapter so the type matches PeftModel
        # and we can load adapters later dynamically.
        if not first_adapter_loaded:
            logger.info("No saved adapters found. Creating a placeholder PEFT wrapper.")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
            )
            # PEFT model with random weights (disabled by default so it runs like the base model)
            self._model = get_peft_model(base_model, lora_config)
            self._model.disable_adapter()
            self._active_adapter = "base"
        else:
            # Set active adapter to the first loaded one
            self._active_adapter = list(self._loaded_adapters.keys())[0]
            self._model.set_adapter(self._active_adapter)
            logger.info("Active adapter set to '%s'.", self._active_adapter)

        self._model.eval()
        self._is_loaded = True
        logger.info("LoRA pipeline initialization complete.")

    # -------------------------------------------------------------- #
    #  Adapter Management                                              #
    # -------------------------------------------------------------- #

    def get_loaded_adapters(self) -> List[str]:
        """Return a list of all loaded adapters including 'base'."""
        adapters = ["base"] + list(self._loaded_adapters.keys())
        return adapters

    def switch_adapter(self, adapter_name: str) -> None:
        """Switch the active LoRA adapter."""
        if not self._is_loaded:
            self.load()

        if adapter_name == "base":
            if isinstance(self._model, PeftModel):
                self._model.disable_adapter()
            self._active_adapter = "base"
            logger.info("Disabled all adapters. Model is now in Base Model mode.")
            return

        if adapter_name not in self._loaded_adapters:
            # Try to see if it is in the adapters directory and load it dynamically
            potential_dir = ADAPTERS_DIR / adapter_name
            if potential_dir.exists() and (potential_dir / "adapter_config.json").exists():
                self.load_new_adapter(str(potential_dir), adapter_name)
            else:
                raise ValueError(f"Adapter '{adapter_name}' is not loaded and does not exist in {ADAPTERS_DIR}")

        if isinstance(self._model, PeftModel):
            self._model.enable_adapter()
            self._model.set_adapter(adapter_name)
            self._active_adapter = adapter_name
            logger.info("Switched active adapter to '%s'.", adapter_name)

    def load_new_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Load a new adapter dynamically at runtime."""
        if not self._is_loaded:
            self.load()

        if adapter_name in self._loaded_adapters:
            logger.info("Adapter '%s' already loaded. Switching instead.", adapter_name)
            self.switch_adapter(adapter_name)
            return

        logger.info("Loading adapter '%s' dynamically from %s...", adapter_name, adapter_path)
        try:
            if isinstance(self._model, PeftModel):
                self._model.load_adapter(adapter_path, adapter_name)
                self._loaded_adapters[adapter_name] = adapter_path
                self.switch_adapter(adapter_name)
            else:
                # Base model wasn't wrapped in PEFT yet, wrap it now
                self._model = PeftModel.from_pretrained(
                    self._model,
                    adapter_path,
                    adapter_name=adapter_name,
                )
                self._loaded_adapters[adapter_name] = adapter_path
                self._active_adapter = adapter_name
                logger.info("Wrapped base model in PEFT and loaded adapter '%s'.", adapter_name)
        except Exception as e:
            logger.error("Failed to load new adapter dynamically: %s", e)
            raise e

    # -------------------------------------------------------------- #
    #  Inference                                                       #
    # -------------------------------------------------------------- #

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        adapter_name: Optional[str] = None,
    ) -> str:
        """Generate text. Switch adapter beforehand if specified."""
        if not self._is_loaded:
            self.load()

        if adapter_name:
            self.switch_adapter(adapter_name)

        # Tokenize inputs
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                generation_config=gen_config,
            )

        # Decode newly generated tokens only
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    # -------------------------------------------------------------- #
    #  Clean Memory Utility                                            #
    # -------------------------------------------------------------- #

    def unload(self) -> None:
        """Unload model from RAM/VRAM to free up resources."""
        if not self._is_loaded:
            return
        logger.info("Unloading LoRA model pipelines to free memory...")
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._loaded_adapters.clear()
        self._active_adapter = "base"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def active_adapter(self) -> str:
        return self._active_adapter
