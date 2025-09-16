"""
Core model handler for Meditron3-Qwen2.5
Handles model loading, inference, and memory management
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TextStreamer
)
from typing import Optional, Dict, List, Any, Union, Generator
from pathlib import Path
import sys
from loguru import logger
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings, get_project_root


class MeditronModel:
    """
    Meditron3-Qwen2.5 model handler with optimized inference
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(get_project_root() / settings.model.model_path)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
        
        # Setup logging
        logger.add(
            get_project_root() / "logs" / "model.log",
            level=settings.logging.log_level,
            rotation="10 MB"
        )
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for memory optimization"""
        if settings.model.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif settings.model.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def load_model(self) -> bool:
        """
        Load the Meditron3-Qwen2.5 model and tokenizer
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("🚀 Loading Meditron3-Qwen2.5 model...")
            
            # Check if model path exists
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.error(f"❌ Model path does not exist: {model_path}")
                return False
            
            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                logger.warning("⚠️ Using CPU (slow inference)")
            
            # Load tokenizer
            logger.info("📚 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✅ Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
            
            # Prepare model loading arguments
            model_kwargs = {
                "torch_dtype": getattr(torch, settings.model.torch_dtype),
                "device_map": settings.model.device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Add quantization if specified
            quantization_config = self._get_quantization_config()
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                logger.info("🔧 Using quantization for memory optimization")
            
            # Load model
            logger.info("🧠 Loading model (this may take a while)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("✅ Model loaded successfully")
            self.is_loaded = True
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"📊 Total parameters: {total_params:,}")
            logger.info(f"🔧 Trainable parameters: {trainable_params:,}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty
            stream: Whether to stream the response
        
        Returns:
            Generated response string or generator for streaming
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use default values from settings if not provided
        max_new_tokens = max_new_tokens or 512
        temperature = temperature or settings.model.temperature
        top_p = top_p or settings.model.top_p
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt",
                add_special_tokens=True
            )
            
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Prepare generation arguments
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if stream:
                return self._generate_stream(inputs, **generation_kwargs)
            else:
                return self._generate_complete(inputs, **generation_kwargs)
                
        except Exception as e:
            logger.error(f"❌ Generation failed: {str(e)}")
            raise
    
    def _generate_complete(self, inputs: torch.Tensor, **kwargs) -> str:
        """Generate complete response"""
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                **kwargs
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):],  # Only new tokens
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _generate_stream(self, inputs: torch.Tensor, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response"""
        # Note: This is a simplified streaming implementation
        # For production, consider using transformers' TextStreamer or similar
        
        with torch.no_grad():
            # For streaming, we generate token by token
            generated_ids = inputs[0].tolist()
            max_new_tokens = kwargs.get("max_new_tokens", 512)
            
            for _ in range(max_new_tokens):
                # Get logits for next token
                with torch.no_grad():
                    outputs = self.model(torch.tensor([generated_ids]).to(self.device))
                    logits = outputs.logits[0, -1, :]
                
                # Apply temperature
                if kwargs.get("temperature", 1.0) != 1.0:
                    logits = logits / kwargs["temperature"]
                
                # Apply top-k filtering
                if kwargs.get("top_k", 0) > 0:
                    top_k = kwargs["top_k"]
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if kwargs.get("top_p", 1.0) < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > kwargs["top_p"]
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if kwargs.get("do_sample", True):
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = torch.argmax(logits).item()
                
                # Check for EOS
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token)
                
                # Decode and yield new token
                new_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                yield new_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **generation_kwargs
    ) -> str:
        """
        Chat interface with conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **generation_kwargs: Generation parameters
        
        Returns:
            Assistant's response
        """
        # Format messages into a prompt
        # This is a simplified implementation - adjust based on model's chat format
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"Human: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        
        return self.generate_response(prompt, **generation_kwargs)
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.is_loaded = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("🗑️ Model unloaded and memory cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_path": self.model_path,
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
        }
        
        if torch.cuda.is_available() and self.device == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
            })
        
        return info


# Global model instance
model_instance = None


def get_model() -> MeditronModel:
    """Get or create global model instance"""
    global model_instance
    if model_instance is None:
        model_instance = MeditronModel()
    return model_instance


if __name__ == "__main__":
    # Test the model
    model = MeditronModel()
    
    if model.load_model():
        print("✅ Model loaded successfully")
        
        # Test generation
        test_prompt = "What is the function of brain extracellular space?"
        response = model.generate_response(test_prompt, max_new_tokens=100)
        print(f"💬 Response: {response}")
        
        # Print model info
        info = model.get_model_info()
        print(f"📊 Model info: {info}")
        
        model.unload_model()
    else:
        print("❌ Failed to load model")