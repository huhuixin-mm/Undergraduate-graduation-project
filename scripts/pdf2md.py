import os
import subprocess
import glob
import shutil
from pathlib import Path
import torch

def process_files(input_dir, output_dir, device="cuda", language="en", source="modelscope", overwrite=False):
    """
    批量处理input目录下的文件
    
    参数:
    input_dir: 输入目录路径
    output_dir: 输出目录路径
    device: 使用的设备 (cuda/cpu)
    language: 语言设置
    source: 模型来源
    overwrite: 是否覆盖已存在的输出目录
    """
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的文件扩展名（根据mineru支持的文件类型调整）
    supported_extensions = ['*.pdf', '*.txt', '*.doc', '*.docx', '*.jpg', '*.jpeg', '*.png']
    
    # 打印具体的物理设备名称
    if device.startswith("cuda"):
        try:
            if torch.cuda.is_available():
                device_index = 0
                if ":" in device:
                    device_index = int(device.split(":")[1])
                device_name = torch.cuda.get_device_name(device_index)
                print(f"使用的设备: {device} ({device_name})")
            else:
                print(f"使用的设备: {device} (CUDA 不可用)")
        except ImportError:
            print(f"使用的设备: {device} (未安装 torch，无法获取具体设备名称)")
    else:
        print(f"使用的设备: {device}")

    # 获取所有支持的文件
    input_files = []
    for ext in supported_extensions:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not input_files:
        print(f"在目录 {input_dir} 中没有找到支持的文件")
        return
    
    total_files = len(input_files)
    print(f"找到 {total_files} 个文件需要处理:")
    
    # 只列出前5个文件
    for file in input_files[:5]:
        print(f"  - {file}")
    if total_files > 5:
        print(f"  - ... (以及其他 {total_files - 5} 个文件)")
    
    # 批量处理文件
    success_count = 0
    failed_files = []
    
    for input_file in input_files:
        try:
            # 获取文件名（不含扩展名）用于输出目录
            filename = Path(input_file).stem
            file_output_dir = output_dir

            # 用于覆盖/跳过判断：MinerU会在输出目录下创建以文件名为名的子目录
            expected_dir = os.path.join(output_dir, filename)
            if os.path.exists(expected_dir):
                if not overwrite:
                    print(f"跳过 {input_file}，目标目录已存在（{expected_dir}）")
                    continue
                print(f"覆盖已存在的目录: {expected_dir}")
                shutil.rmtree(expected_dir)

            os.makedirs(file_output_dir, exist_ok=True)
            
            # 构建命令
            cmd = [
                'mineru', '-p', input_file, '-o', file_output_dir,
                '-d', device, '-l', language, '--source', source
            ]
            
            print(f"\n正在处理: {input_file}")
            print(f"命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print(f"✓ 成功处理: {input_file}")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"✗ 处理失败: {input_file}")
            print(f"错误信息: {e.stderr}")
            failed_files.append(input_file)
        except Exception as e:
            print(f"✗ 处理异常: {input_file}")
            print(f"异常信息: {e}")
            failed_files.append(input_file)
    
    # 输出处理结果摘要
    print(f"\n{'='*50}")
    print("处理完成!")
    print(f"成功: {success_count}/{len(input_files)}")
    if failed_files:
        print(f"失败的文件:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = "./data/raw/en"
    OUTPUT_DIR = "./data/processed/markdown/en"
    DEVICE = "cuda:1"
    LANGUAGE = "en"
    SOURCE = "modelscope"
    OVERWRITE = False  # 需要强制重跑时改为 True
    
    # 检查mineru命令是否可用
    try:
        subprocess.run(['mineru', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 未找到mineru命令，请确保已正确安装并添加到PATH环境变量中")
        print("可以尝试运行: export PATH=\"/home/huhuixin/.local/bin:$PATH\"")
        exit(1)
    
    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        exit(1)
    
    # 执行批量处理
    process_files(INPUT_DIR, OUTPUT_DIR, DEVICE, LANGUAGE, SOURCE, overwrite=OVERWRITE)