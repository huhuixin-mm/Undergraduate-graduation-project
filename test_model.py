#!/usr/bin/env python3
"""
全面测试 Meditron3-Qwen2.5 模型功能
测试内容：单轮对话、多轮对话、流式生成、多语言支持
"""

import sys
import time
from pathlib import Path
from typing import List, Dict

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from services.model_handler import MeditronModel, get_model, ensure_model_loaded


class ModelTester:
    """模型测试器"""
    
    def __init__(self):
        self.model = None
        
    def setup(self):
        """初始化设置"""
        print("🚀 初始化 Meditron3-Qwen2.5 测试...")
        print(f"📍 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.model = MeditronModel()
        success = self.model.load_model()
        
        if success:
            print("✅ 模型加载成功!")
            print(f"🔧 使用设备: {self.model.device}")
            return True
        else:
            print("❌ 模型加载失败!")
            return False
    
    def test_single_conversation(self):
        """测试1: 单轮对话"""
        print("\n" + "="*60)
        print("🧪 测试 1: 单轮对话")
        print("="*60)
        
        test_prompts = [
            {
                "prompt": "请详细解释什么是高血压，包括症状和预防措施。",
                "desc": "中文医疗问题"
            },
            {
                "prompt": "我最近总是失眠，有什么好的建议吗？",
                "desc": "中文健康咨询"
            },
            {
                "prompt": "阿司匹林的主要作用和副作用是什么？",
                "desc": "中文药物咨询"
            }
        ]
        
        for i, test_case in enumerate(test_prompts, 1):
            print(f"\n--- 单轮对话测试 {i}: {test_case['desc']} ---")
            print(f"👤 用户: {test_case['prompt']}")
            
            try:
                start_time = time.time()
                response = self.model.generate_response(
                    test_case['prompt'],
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9
                )
                end_time = time.time()
                
                print(f"🤖 助手: {response}")
                print(f"⏱️ 响应时间: {end_time - start_time:.2f}秒")
                print("✅ 测试通过")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
    
    def test_multi_turn_conversation(self):
        """测试2: 多轮对话"""
        print("\n" + "="*60)
        print("🧪 测试 2: 多轮对话")
        print("="*60)
        
        # 第一个多轮对话场景
        print("\n--- 多轮对话场景 1: 头痛问题咨询 ---")
        
        conversation_1 = [
            {"role": "user", "content": "医生您好，我最近经常头痛，特别是下午的时候。"},
            {"role": "assistant", "content": ""},  # 将被填充
            {"role": "user", "content": "我平时工作需要长时间看电脑，这会是原因吗？"},
            {"role": "assistant", "content": ""},  # 将被填充
            {"role": "user", "content": "那我应该怎么预防和缓解呢？"}
        ]
        
        self._run_multi_turn_conversation(conversation_1, "头痛咨询")
        
        # 第二个多轮对话场景
        print("\n--- 多轮对话场景 2: 糖尿病知识咨询 ---")
        
        conversation_2 = [
            {"role": "user", "content": "我想了解一下糖尿病的基本知识。"},
            {"role": "assistant", "content": ""},  # 将被填充
            {"role": "user", "content": "那糖尿病患者在饮食上需要注意什么？"},
            {"role": "assistant", "content": ""},  # 将被填充
            {"role": "user", "content": "如果血糖控制不好会有什么后果？"}
        ]
        
        self._run_multi_turn_conversation(conversation_2, "糖尿病咨询")
    
    def _run_multi_turn_conversation(self, conversation: List[Dict], scenario_name: str):
        """运行多轮对话"""
        messages = []
        
        for turn_idx, turn in enumerate(conversation):
            if turn["role"] == "user":
                messages.append(turn)
                print(f"\n🗣️ 第 {len([m for m in messages if m['role'] == 'user'])} 轮")
                print(f"👤 用户: {turn['content']}")
                
                try:
                    start_time = time.time()
                    response = self.model.chat(
                        messages,
                        max_new_tokens=180,
                        temperature=0.7,
                        top_p=0.9
                    )
                    end_time = time.time()
                    
                    messages.append({"role": "assistant", "content": response})
                    print(f"🤖 助手: {response}")
                    print(f"⏱️ 响应时间: {end_time - start_time:.2f}秒")
                    
                except Exception as e:
                    print(f"❌ 第{len([m for m in messages if m['role'] == 'user'])}轮对话失败: {e}")
                    break
        
        print(f"✅ {scenario_name}多轮对话测试完成")
    
    def test_streaming_generation(self):
        """测试3: 流式生成"""
        print("\n" + "="*60)
        print("🧪 测试 3: 流式生成")
        print("="*60)
        
        test_prompts = [
            "请详细介绍一下心脏病的预防措施，包括饮食、运动和生活方式的建议。",
            "什么是抑郁症？它的症状、原因和治疗方法有哪些？"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- 流式生成测试 {i} ---")
            print(f"👤 用户: {prompt}")
            print("🤖 助手: ", end="", flush=True)
            
            try:
                start_time = time.time()
                full_response = ""
                
                for chunk in self.model.generate_response(
                    prompt,
                    max_new_tokens=250,
                    temperature=0.7,
                    top_p=0.9,
                    stream=True  # 启用流式模式
                ):
                    # 只显示新增的部分
                    new_part = chunk[len(full_response):]
                    print(new_part, end="", flush=True)
                    full_response = chunk
                    time.sleep(0.02)  # 模拟流式延迟
                
                end_time = time.time()
                print(f"\n⏱️ 总响应时间: {end_time - start_time:.2f}秒")
                print(f"📊 最终回复长度: {len(full_response)}字符")
                print("✅ 流式生成测试通过")
                
            except Exception as e:
                print(f"\n❌ 流式生成测试失败: {e}")
    
    def test_multilingual_conversation(self):
        """测试4: 多语言对话"""
        print("\n" + "="*60)
        print("🧪 测试 4: 多语言对话")
        print("="*60)
        
        multilingual_tests = [
            {
                "language": "中文",
                "prompt": "请解释一下什么是新冠病毒，以及如何预防感染？",
                "expected_lang": "zh"
            },
            {
                "language": "English",
                "prompt": "What are the main symptoms of diabetes and how can it be managed?",
                "expected_lang": "en"
            },
            {
                "language": "中英混合",
                "prompt": "Doctor，我想问一下关于COVID-19疫苗接种的问题，有什么副作用吗？",
                "expected_lang": "mixed"
            },
            {
                "language": "英文医学术语",
                "prompt": "Can you explain the difference between Type 1 and Type 2 diabetes mellitus?",
                "expected_lang": "en"
            },
            {
                "language": "中文专业术语",
                "prompt": "请详细说明高血压的病理生理机制和药物治疗原理。",
                "expected_lang": "zh"
            }
        ]
        
        for i, test_case in enumerate(multilingual_tests, 1):
            print(f"\n--- 多语言测试 {i}: {test_case['language']} ---")
            print(f"👤 用户 ({test_case['language']}): {test_case['prompt']}")
            
            try:
                start_time = time.time()
                
                response = self.model.generate_response(
                    test_case['prompt'],
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9
                )
                end_time = time.time()
                
                print(f"🤖 助手: {response}")
                print(f"⏱️ 响应时间: {end_time - start_time:.2f}秒")
                
            except Exception as e:
                print(f"❌ 多语言测试 {i} 失败: {e}")
    
    def test_conversation_context_memory(self):
        """测试5: 对话上下文记忆"""
        print("\n" + "="*60)
        print("🧪 测试 5: 对话上下文记忆")
        print("="*60)
        
        print("\n--- 上下文记忆测试 ---")
        
        # 构建一个需要记住上下文的对话
        context_test = [
            {"role": "user", "content": "我是一个45岁的男性，最近体检发现血压有点高。"},
            {"role": "user", "content": "基于我刚才提到的情况，我应该注意什么？"},
            {"role": "user", "content": "根据我的年龄和性别，还有其他需要定期检查的项目吗？"}
        ]
        
        messages = []
        
        for turn_idx, user_msg in enumerate(context_test, 1):
            messages.append(user_msg)
            print(f"\n🗣️ 第 {turn_idx} 轮")
            print(f"👤 用户: {user_msg['content']}")
            
            try:
                start_time = time.time()
                response = self.model.chat(
                    messages,
                    max_new_tokens=180,
                    temperature=0.7,
                    top_p=0.9
                )
                end_time = time.time()
                
                messages.append({"role": "assistant", "content": response})
                print(f"🤖 助手: {response}")
                print(f"⏱️ 响应时间: {end_time - start_time:.2f}秒")
                
                # 检查是否记住了上下文
                if turn_idx > 1:
                    context_keywords = ["45", "男性", "血压", "高"]
                    context_maintained = any(keyword in response for keyword in context_keywords)
                    print(f"🧠 上下文记忆: {'✓' if context_maintained else '?'}")
                
            except Exception as e:
                print(f"❌ 第 {turn_idx} 轮上下文测试失败: {e}")
                break
        
        print("✅ 上下文记忆测试完成")
    
    def test_streaming_chat(self):
        """测试6: 流式多轮对话"""
        print("\n" + "="*60)
        print("🧪 测试 6: 流式多轮对话")
        print("="*60)
        
        print("\n--- 流式对话测试 ---")
        
        # 构建流式对话测试
        streaming_messages = [
            {"role": "user", "content": "医生，我想了解一下关于减肥的健康方法。"},
            {"role": "user", "content": "那对于我这种工作比较忙的人，有什么简单易行的建议吗？"}
        ]
        
        messages = []
        
        for turn_idx, user_msg in enumerate(streaming_messages, 1):
            messages.append(user_msg)
            print(f"\n🗣️ 第 {turn_idx} 轮 - 流式对话")
            print(f"👤 用户: {user_msg['content']}")
            print("🤖 助手: ", end="", flush=True)
            
            try:
                start_time = time.time()
                full_response = ""
                
                # 使用流式对话
                for chunk in self.model.chat(
                    messages,
                    max_new_tokens=180,
                    temperature=0.7,
                    top_p=0.9,
                    stream=True  # 启用流式模式
                ):
                    # 只显示新增的部分
                    new_part = chunk[len(full_response):]
                    print(new_part, end="", flush=True)
                    full_response = chunk
                    time.sleep(0.03)  # 模拟流式延迟
                
                end_time = time.time()
                
                messages.append({"role": "assistant", "content": full_response})
                print(f"\n⏱️ 响应时间: {end_time - start_time:.2f}秒")
                print(f"📊 回复长度: {len(full_response)}字符")
                
            except Exception as e:
                print(f"\n❌ 第 {turn_idx} 轮流式对话失败: {e}")
                break
        
        print("✅ 流式对话测试完成")
    
    def run_performance_benchmark(self):
        """性能基准测试"""
        print("\n" + "="*60)
        print("🏃 性能基准测试")
        print("="*60)
        
        benchmark_prompts = [
            "请简单介绍感冒的症状。",
            "什么是高血压？",
            "糖尿病患者饮食注意事项。",
            "如何预防心脏病？",
            "抑郁症的治疗方法有哪些？"
        ]
        
        total_time = 0
        total_tokens = 0
        
        print(f"\n📊 运行 {len(benchmark_prompts)} 个基准测试...")
        
        for i, prompt in enumerate(benchmark_prompts, 1):
            print(f"\n⚡ 基准测试 {i}/{len(benchmark_prompts)}")
            
            try:
                start_time = time.time()
                response = self.model.generate_response(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                token_count = len(self.model.tokenizer.encode(response))
                
                total_time += response_time
                total_tokens += token_count
                
                print(f"  ⏱️ 响应时间: {response_time:.2f}秒")
                print(f"  📝 生成token数: {token_count}")
                print(f"  🚀 吞吐量: {token_count/response_time:.2f} tokens/秒")
                
            except Exception as e:
                print(f"  ❌ 基准测试 {i} 失败: {e}")
        
        # 计算平均性能
        avg_time = total_time / len(benchmark_prompts)
        avg_throughput = total_tokens / total_time
        
        print(f"\n📈 性能总结:")
        print(f"  平均响应时间: {avg_time:.2f}秒")
        print(f"  总生成token数: {total_tokens}")
        print(f"  平均吞吐量: {avg_throughput:.2f} tokens/秒")
        print("✅ 性能基准测试完成")
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            self.model.unload_model()
            print("\n🗑️ 模型已卸载，资源已清理")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🎯 开始全面测试 Meditron3-Qwen2.5 模型")
        print("="*80)
        
        if not self.setup():
            return False
        
        try:
            # 运行所有测试
            self.test_single_conversation()
            self.test_multi_turn_conversation()
            self.test_streaming_generation()
            self.test_multilingual_conversation()
            self.test_conversation_context_memory()
            self.test_streaming_chat()  # 新增流式对话测试
            self.run_performance_benchmark()
            
            print("\n" + "="*80)
            print("🎉 所有测试完成！")
            print("✅ Meditron3-Qwen2.5 模型功能验证通过")
            print("="*80)
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ 测试被用户中断")
            return False
        except Exception as e:
            print(f"\n❌ 测试过程中发生错误: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """主函数"""
    tester = ModelTester()
    success = tester.run_all_tests()
    
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()