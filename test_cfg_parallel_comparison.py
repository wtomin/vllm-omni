"""
测试脚本：对比开启和不开启 cfg-parallel 时的输出差异
验证 cfg-parallel 是否为无损加速（pixel 差异应该为零）

通过调用 text_to_image.py 脚本生成图像，然后对比像素差异
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"


def run_text_to_image(cfg_parallel_size: int, output_path: str, **kwargs) -> Path:
    """
    调用 text_to_image.py 脚本生成图像
    
    Args:
        cfg_parallel_size: CFG 并行大小 (1=不开启, 2=开启)
        output_path: 输出图像路径
        **kwargs: 其他参数（model, prompt, seed, height, width, 等）
    
    Returns:
        输出图像的 Path 对象
    """
    text_to_image_script = REPO_ROOT / "examples" / "offline_inference" / "text_to_image" / "text_to_image.py"
    
    if not text_to_image_script.exists():
        raise FileNotFoundError(f"找不到 text_to_image.py: {text_to_image_script}")
    
    # 构建命令行参数
    cmd = [
        sys.executable,
        str(text_to_image_script),
        "--cfg-parallel-size", str(cfg_parallel_size),
        "--output", output_path,
    ]
    
    # 添加其他参数
    for key, value in kwargs.items():
        param_name = "--" + key.replace("_", "-")
        cmd.extend([param_name, str(value)])
    
    print(f"运行命令: {' '.join(cmd)}")
    
    # 运行脚本
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"text_to_image.py 运行失败，返回码: {result.returncode}")
    
    print(result.stdout)
    
    output = Path(output_path)
    if not output.exists():
        raise FileNotFoundError(f"生成的图像不存在: {output}")
    
    return output


def test_cfg_parallel_lossless():
    """
    通过调用 text_to_image.py 脚本，对比开启和不开启 cfg-parallel 时的输出，验证是否无损。
    """
    model_name = "riverclouds/qwen_image_random"
    prompt = "a photo of a cat sitting on a laptop keyboard"
    seed = 42
    height = 256
    width = 256
    num_inference_steps = 4
    guidance_scale = 3.0  # 必须 > 1.0 才能触发 CFG
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    img_with_cfg_path = output_dir / "cfg_parallel_enabled.png"
    img_without_cfg_path = output_dir / "cfg_parallel_disabled.png"
    
    try:
        print("=" * 80)
        print("CFG-Parallel 无损测试")
        print("=" * 80)
        print(f"测试模型: {model_name}")
        print(f"提示词: {prompt}")
        print(f"分辨率: {width}x{height}")
        print(f"推理步数: {num_inference_steps}")
        print(f"Guidance Scale: {guidance_scale}")
        print(f"随机种子: {seed}")
        print("=" * 80)
        
        # 测试参数
        common_params = {
            "model": model_name,
            "prompt": prompt,
            "seed": seed,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
        }
        
        # [1/2] 生成开启 cfg-parallel 的图像
        print("\n[1/2] 生成图像 (cfg-parallel-size=2, 开启 CFG 并行)...")
        print("-" * 80)
        run_text_to_image(
            cfg_parallel_size=2,
            output_path=str(img_with_cfg_path),
            **common_params
        )
        
        # [2/2] 生成不开启 cfg-parallel 的图像
        print("\n[2/2] 生成图像 (cfg-parallel-size=1, 不开启 CFG 并行)...")
        print("-" * 80)
        run_text_to_image(
            cfg_parallel_size=1,
            output_path=str(img_without_cfg_path),
            **common_params
        )
        
        # 加载图像并对比
        print("\n" + "=" * 80)
        print("对比结果分析")
        print("=" * 80)
        
        image_with_cfg = Image.open(img_with_cfg_path)
        image_without_cfg = Image.open(img_without_cfg_path)
        
        img_array_with_cfg = np.array(image_with_cfg)
        img_array_without_cfg = np.array(image_without_cfg)
        
        # 验证形状一致
        assert img_array_with_cfg.shape == img_array_without_cfg.shape, \
            f"图像形状不匹配: {img_array_with_cfg.shape} vs {img_array_without_cfg.shape}"
        print(f"✓ 图像形状一致: {img_array_with_cfg.shape}")
        
        # 计算像素差异
        pixel_diff = np.abs(img_array_with_cfg.astype(np.float32) - img_array_without_cfg.astype(np.float32))
        max_diff = np.max(pixel_diff)
        mean_diff = np.mean(pixel_diff)
        num_different_pixels = np.sum(pixel_diff > 0)
        total_pixels = pixel_diff.size
        
        print(f"\n像素差异统计:")
        print(f"  - 最大像素差异: {max_diff}")
        print(f"  - 平均像素差异: {mean_diff:.6f}")
        print(f"  - 不同像素数量: {num_different_pixels} / {total_pixels}")
        print(f"  - 不同像素比例: {num_different_pixels / total_pixels * 100:.6f}%")
        
        print(f"\n✓ 已保存图像:")
        print(f"  - cfg-parallel-size=2: {img_with_cfg_path}")
        print(f"  - cfg-parallel-size=1: {img_without_cfg_path}")
        
        # 如果有差异，保存差异图
        if max_diff > 0:
            diff_path = output_dir / "pixel_difference.png"
            # 将差异图归一化到 0-255 范围
            diff_normalized = (pixel_diff / max_diff * 255).astype(np.uint8)
            if len(diff_normalized.shape) == 3 and diff_normalized.shape[2] == 3:
                # RGB 图像，转换为灰度图以便查看
                diff_gray = np.mean(diff_normalized, axis=2).astype(np.uint8)
                diff_img = Image.fromarray(diff_gray, mode='L')
            else:
                diff_img = Image.fromarray(diff_normalized)
            diff_img.save(diff_path)
            print(f"  - 差异可视化: {diff_path}")
        
        # 最终断言：cfg-parallel 应该是无损的
        print("\n" + "=" * 80)
        if max_diff == 0:
            print("✅ 测试通过: cfg-parallel 是无损加速，像素差异为零！")
            print("=" * 80)
        else:
            print("❌ 测试失败: cfg-parallel 存在像素差异！")
            print("=" * 80)
            raise AssertionError(
                f"cfg-parallel 应该是无损加速，但发现最大像素差异为 {max_diff}"
            )
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_cfg_parallel_lossless()
