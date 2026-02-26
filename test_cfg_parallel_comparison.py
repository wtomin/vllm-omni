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


def _run_single_steps_comparison(
    steps: int,
    output_dir: Path,
    common_params: dict,
    tolerance: float,
) -> tuple[bool, float]:
    """
    对单个 num_inference_steps 值运行 cfg-parallel 对比测试。

    Returns:
        (passed, max_diff)
    """
    print("\n" + "=" * 80)
    print(f"num_inference_steps = {steps}")
    print("=" * 80)

    img_with_cfg_path = output_dir / f"steps{steps}_cfg_parallel_enabled.png"
    img_without_cfg_path = output_dir / f"steps{steps}_cfg_parallel_disabled.png"

    params = {**common_params, "num_inference_steps": steps}

    print(f"\n[1/2] 生成图像 (cfg-parallel-size=2, 开启 CFG 并行)...")
    print("-" * 80)
    run_text_to_image(cfg_parallel_size=2, output_path=str(img_with_cfg_path), **params)

    print(f"\n[2/2] 生成图像 (cfg-parallel-size=1, 不开启 CFG 并行)...")
    print("-" * 80)
    run_text_to_image(cfg_parallel_size=1, output_path=str(img_without_cfg_path), **params)

    image_with_cfg = Image.open(img_with_cfg_path)
    image_without_cfg = Image.open(img_without_cfg_path)

    img_array_with_cfg = np.array(image_with_cfg)
    img_array_without_cfg = np.array(image_without_cfg)

    assert img_array_with_cfg.shape == img_array_without_cfg.shape, (
        f"图像形状不匹配: {img_array_with_cfg.shape} vs {img_array_without_cfg.shape}"
    )
    print(f"✓ 图像形状一致: {img_array_with_cfg.shape}")

    # 归一化到 [0, 255] 范围后计算像素差异
    arr_with = img_array_with_cfg.astype(np.float32)
    arr_without = img_array_without_cfg.astype(np.float32)
    max_val_with = arr_with.max()
    max_val_without = arr_without.max()
    arr_with_norm = arr_with / max_val_with * 255.0 if max_val_with > 0 else arr_with
    arr_without_norm = arr_without / max_val_without * 255.0 if max_val_without > 0 else arr_without

    pixel_diff = np.abs(arr_with_norm - arr_without_norm)
    max_diff = float(np.max(pixel_diff))
    mean_diff = float(np.mean(pixel_diff))
    num_different_pixels = int(np.sum(pixel_diff > 0))
    total_pixels = pixel_diff.size

    print(f"\n归一化后像素差异统计 (各自归一化到 0-255):")
    print(f"  - 最大像素差异: {max_diff:.6f}")
    print(f"  - 平均像素差异: {mean_diff:.6f}")
    print(f"  - 不同像素数量: {num_different_pixels} / {total_pixels}")
    print(f"  - 不同像素比例: {num_different_pixels / total_pixels * 100:.6f}%")

    print(f"\n✓ 已保存图像:")
    print(f"  - cfg-parallel-size=2: {img_with_cfg_path}")
    print(f"  - cfg-parallel-size=1: {img_without_cfg_path}")

    if max_diff > 0:
        diff_path = output_dir / f"steps{steps}_pixel_difference.png"
        diff_normalized = (pixel_diff / max_diff * 255).astype(np.uint8)
        if len(diff_normalized.shape) == 3 and diff_normalized.shape[2] == 3:
            diff_gray = np.mean(diff_normalized, axis=2).astype(np.uint8)
            diff_img = Image.fromarray(diff_gray, mode='L')
        else:
            diff_img = Image.fromarray(diff_normalized)
        diff_img.save(diff_path)
        print(f"  - 差异可视化: {diff_path}")

    passed = max_diff <= tolerance
    if passed:
        print(f"\n✅ steps={steps} 通过: 归一化最大像素差异 {max_diff:.6f} <= {tolerance}")
    else:
        print(f"\n❌ steps={steps} 失败: 归一化最大像素差异 {max_diff:.6f} > {tolerance}")

    return passed, max_diff


def test_cfg_parallel_lossless():
    """
    遍历多个 num_inference_steps，对比开启和不开启 cfg-parallel 时的输出，验证是否无损。
    """
    model_name = "Qwen/Qwen-Image-2512"
    prompt = "'a photo of a cat sitting on a laptop keyboard'"
    seed = 42
    height = 256
    width = 256
    num_inference_steps_list = [2, 4, 8, 16]
    guidance_scale = 3.0  # 必须 > 1.0 才能触发 CFG
    # tolerance 单位为归一化后的像素值（0-255 范围），仅允许浮点精度误差
    tolerance = 1e-3

    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("CFG-Parallel 无损测试")
    print("=" * 80)
    print(f"测试模型: {model_name}")
    print(f"提示词: {prompt}")
    print(f"分辨率: {width}x{height}")
    print(f"推理步数列表: {num_inference_steps_list}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"随机种子: {seed}")
    print("=" * 80)

    common_params = {
        "model": model_name,
        "prompt": prompt,
        "negative_prompt": "'ugly, unclear'",
        "seed": seed,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }

    failures: list[tuple[int, float]] = []

    for steps in num_inference_steps_list:
        try:
            passed, max_diff = _run_single_steps_comparison(
                steps=steps,
                output_dir=output_dir,
                common_params=common_params,
                tolerance=tolerance,
            )
            if not passed:
                failures.append((steps, max_diff))
        except Exception as e:
            print(f"\n❌ steps={steps} 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            failures.append((steps, float("nan")))

    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    for steps in num_inference_steps_list:
        failed_steps = [s for s, _ in failures]
        status = "❌ 失败" if steps in failed_steps else "✅ 通过"
        print(f"  steps={steps:>3}: {status}")

    if failures:
        print("=" * 80)
        details = ", ".join(
            f"steps={s} (max_diff={d:.6f})" for s, d in failures
        )
        raise AssertionError(
            f"cfg-parallel 在以下配置存在超出阈值的像素差异: {details}"
        )

    print("✅ 所有推理步数测试通过: cfg-parallel 是无损加速！")
    print("=" * 80)


if __name__ == "__main__":
    test_cfg_parallel_lossless()
