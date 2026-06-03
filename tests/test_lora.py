"""
test_lora.py — Unit tests for the LoRAPipeline
==============================================

Tests device loading configs, device overrides, fallback choices,
dynamic adapter registration, and properties on LoRAPipeline with mock imports.
"""

import pytest
from unittest.mock import MagicMock, patch
from peft import PeftModel
from src.lora_pipeline import LoRAPipeline

class DummyPeftModel(PeftModel):
    def load_adapter(self, *args, **kwargs):
        pass
    def set_adapter(self, *args, **kwargs):
        pass
    def disable_adapter(self, *args, **kwargs):
        pass
    def enable_adapter(self, *args, **kwargs):
        pass

@pytest.fixture
def mock_transformers():
    """Mock the heavy HuggingFace model and tokenizer loads."""
    with patch("src.lora_pipeline.AutoTokenizer.from_pretrained") as mock_tok_load, \
         patch("src.lora_pipeline.AutoModelForCausalLM.from_pretrained") as mock_model_load, \
         patch("src.lora_pipeline.BitsAndBytesConfig") as mock_bnb, \
         patch("peft.PeftModel.from_pretrained") as mock_peft_from_pretrained, \
         patch("src.lora_pipeline.get_peft_model") as mock_get_peft:
         
        # Mock tokenizer setup
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "</s>"
        mock_tok.pad_token_id = 0
        mock_tok_load.return_value = mock_tok

        # Mock model setup
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_load.return_value = mock_model
        
        # Mock Peft wrapper
        mock_peft_model = MagicMock(spec=DummyPeftModel)
        mock_peft_model.eval.return_value = None
        mock_peft_from_pretrained.return_value = mock_peft_model
        mock_get_peft.return_value = mock_peft_model
        
        yield {
            "tokenizer": mock_tok,
            "model": mock_model,
            "peft": mock_peft_model,
            "peft_class": mock_peft_from_pretrained
        }


def test_device_fallback_detection():
    """Test device selection and light fallback on CPU environments."""
    # Run pipeline initialization with cpu forcing
    pipeline = LoRAPipeline(base_model_name="mistralai/Mistral-7B-Instruct-v0.1", device="cpu")
    # Should automatically fall back to TinyLlama to save CPU RAM
    assert "TinyLlama" in pipeline.base_model_name


def test_lazy_loading(mock_transformers):
    """Test that model loading is deferred until explicitly loaded."""
    pipeline = LoRAPipeline(device="cpu")
    assert not pipeline.is_loaded
    
    # Load model
    pipeline.load()
    assert pipeline.is_loaded
    assert pipeline.active_adapter == "base"


def test_adapter_switching_base(mock_transformers):
    """Test adapter switching and disabling."""
    pipeline = LoRAPipeline(device="cpu")
    pipeline.load()
    
    # Disabling adapter switches back to base mode
    pipeline.switch_adapter("base")
    assert pipeline.active_adapter == "base"


def test_dynamic_adapter_loading(mock_transformers):
    """Test dynamic registration of new adapters at runtime."""
    pipeline = LoRAPipeline(device="cpu")
    pipeline.load()

    # Stub loaded adapters dict
    pipeline._loaded_adapters = {"my_adapter": "/path/to/adapter"}
    
    # Switch to loaded adapter
    pipeline.switch_adapter("my_adapter")
    assert pipeline.active_adapter == "my_adapter"


def test_unload_memory(mock_transformers):
    """Test model unload and memory cleanup."""
    pipeline = LoRAPipeline(device="cpu")
    pipeline.load()
    assert pipeline.is_loaded
    
    pipeline.unload()
    assert not pipeline.is_loaded
    assert len(pipeline._loaded_adapters) == 0
