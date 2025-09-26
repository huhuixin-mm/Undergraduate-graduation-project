"""
Optimized model handler for Meditron3-Qwen2.5
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings, get_project_root


class MeditronModel:
    """Meditron3-Qwen2.5模型处理器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(get_project_root() / settings.model.model_path)
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device()
        self.is_loaded = False
        
        # 设置日志
        logger.add(
            get_project_root() / "logs" / "model.log",
            level=settings.logging.log_level,
            rotation="10 MB"
        )
    
    def _setup_device(self) -> str:
        """设置并返回最优GPU设备"""
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA不可用，使用CPU（推理较慢）")
            return "cpu"
        return "cuda:0"
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """获取量化配置以优化显存使用"""
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
        """加载模型和分词器"""
        try:
            logger.info("🚀 加载 Meditron3-Qwen2.5 模型...")
            
            # 检查模型路径
            if not Path(self.model_path).exists():
                logger.error(f"❌ 模型路径不存在: {self.model_path}")
                return False
            
            # 加载分词器
            logger.info("📚 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 设置填充token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✅ 分词器加载完成: vocab_size={self.tokenizer.vocab_size}")
            
            # 准备模型加载参数
            model_kwargs = {
                "dtype": getattr(torch, settings.model.dtype),
                "device_map": self.device if self.device.startswith("cuda:") else "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # 添加量化配置
            quantization_config = self._get_quantization_config()
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                logger.info("🔧 使用量化优化显存")
            
            # 加载模型
            logger.info("🧠 加载模型（这可能需要一些时间）...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            self.model.eval()
            self.is_loaded = True
            
            # 显示模型信息
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ 模型加载成功")
            logger.info(f"📊 总参数量: {total_params:,}")
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ):
        """
        生成回复 - 支持普通和流式输出
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            stream: 是否使用流式输出 (默认False)
            **kwargs: 其他生成参数
        
        Returns:
            str: 普通模式返回完整回复字符串
            Generator: 流式模式返回生成器，逐步输出文本
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 设置默认参数
        max_new_tokens = max_new_tokens or 512
        temperature = temperature or settings.model.temperature
        top_p = top_p or settings.model.top_p
        repetition_penalty = repetition_penalty or settings.model.repetition_penalty
        
        # 准备生成参数
        generation_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            **kwargs  # 允许传入其他参数
        }
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=settings.model.max_length
            )
            
            # 移动到设备
            if self.device.startswith("cuda"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_params,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码回复
            response = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            ).strip()
            
            # 调试信息
            if len(response) == 0:
                logger.warning(f"⚠️ 生成了空回复，输入长度: {len(inputs['input_ids'][0])}, 输出长度: {len(outputs[0])}")
                logger.warning(f"⚠️ 生成参数: {generation_params}")
                # 尝试不跳过特殊token的解码
                raw_response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):])
                logger.warning(f"⚠️ 原始输出: '{raw_response}'")
            
            # 根据stream参数决定返回方式
            if stream:
                return self._create_stream_generator(response)
            else:
                return response
            
        except Exception as e:
            logger.error(f"❌ 生成回复时出错: {e}")
            raise
    
    def _create_stream_generator(self, full_response: str):
        """创建流式生成器"""
        # 按词分割进行流式输出
        words = full_response.split()
        current_text = ""
        
        for word in words:
            current_text += word + " "
            yield current_text.strip()
        
        # 确保最后返回完整响应
        if current_text.strip() != full_response:
            yield full_response
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        """
        多轮对话接口
        
        Args:
            messages: 对话消息列表
            stream: 是否使用流式输出 (默认False)
            **kwargs: 其他生成参数
        
        Returns:
            str: 普通模式返回完整回复字符串
            Generator: 流式模式返回生成器，逐步输出文本
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 构建对话提示
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                prompt_parts.append(f"用户: {content}")
            elif role == "assistant":
                prompt_parts.append(f"助手: {content}")
            elif role == "system":
                prompt_parts.append(f"系统: {content}")
        
        prompt_parts.append("助手: ")
        prompt = "\n".join(prompt_parts)
        
        return self.generate_response(prompt, stream=stream, **kwargs)
    
    def unload_model(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("🗑️ 模型已卸载，显存已释放")
    
    def __del__(self):
        """析构函数：自动清理资源"""
        self.unload_model()


# 全局模型实例
_global_model: Optional[MeditronModel] = None


def get_model() -> MeditronModel:
    """获取全局模型实例（单例模式）"""
    global _global_model
    if _global_model is None:
        _global_model = MeditronModel()
    return _global_model


def ensure_model_loaded() -> bool:
    """确保模型已加载"""
    model = get_model()
    if not model.is_loaded:
        return model.load_model()
    return True


# 主函数用于测试
if __name__ == "__main__":
    logger.info("🧪 开始模型测试...")
    
    model = MeditronModel()
    if model.load_model():
        logger.info("✅ 模型加载成功，开始测试推理...")
        
        test_prompt = "请简单介绍一下ECS (Extracellular Space) 的作用。"
        response = model.generate_response(test_prompt, max_new_tokens=100)
        
        logger.info(f"📝 测试提示: {test_prompt}")
        logger.info(f"🤖 模型回复: {response}")
        
        model.unload_model()
        logger.info("🎉 测试完成！")
    else:
        logger.error("❌ 模型加载失败！")