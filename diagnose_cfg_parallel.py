"""
诊断脚本：详细分析 cfg-parallel 在不同步数下的行为
输出每一步的中间结果，帮助定位问题根源
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"


def generate_and_extract_image(
    model_name,
    prompt,
    negative_prompt,
    seed,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    true_cfg_scale,
    cfg_parallel_size,
):
    """生成图像并返回 numpy 数组"""
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    
    parallel_config = DiffusionParallelConfig(
        cfg_parallel_size=cfg_parallel_size,
    )
    
    m = Omni(
        model=model_name,
        parallel_config=parallel_config,
    )
    
    try:
        outputs = m.generate(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            },
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                num_outputs_per_prompt=1,
            ),
        )
        
        first_output = outputs[0]
        assert first_output.final_output_type == "image"
        req_out = first_output.request_output[0]
        assert isinstance(req_out, OmniRequestOutput)
        images = req_out.images
        assert len(images) == 1
        
        return np.array(images[0])
    finally:
        m.close()


def compare_outputs(img1, img2, name1, name2):
    """对比两个图像的差异"""
    assert img1.shape == img2.shape, f"形状不匹配: {img1.shape} vs {img2.shape}"
    
    pixel_diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    max_diff = np.max(pixel_diff)
    mean_diff = np.mean(pixel_diff)
    num_different = np.sum(pixel_diff > 0)
    total_pixels = pixel_diff.size
    
    print(f"\n对比 {name1} vs {name2}:")
    print(f"  - 最大像素差异: {max_diff}")
    print(f"  - 平均像素差异: {mean_diff:.6f}")
    print(f"  - 不同像素数: {num_different} / {total_pixels}")
    print(f"  - 不同像素比例: {num_different / total_pixels * 100:.2f}%")
    
    return max_diff, mean_diff, num_different, total_pixels


def diagnose_cfg_parallel():
    """
    诊断不同推理步数下 cfg-parallel 的行为
    """
    model_name = "riverclouds/qwen_image_random"
    prompt = "a photo of a cat sitting on a laptop keyboard"
    negative_prompt = "ugly, unclear"
    seed = 42
    height = 256
    width = 256
    guidance_scale = 3.0
    true_cfg_scale = 4.0
    
    # 测试不同的步数
    test_steps = [2, 4, 6, 8, 10, 12]
    
    print("=" * 100)
    print("CFG-Parallel 诊断测试")
    print("=" * 100)
    print(f"模型: {model_name}")
    print(f"提示词: {prompt}")
    print(f"分辨率: {width}x{height}")
    print(f"Negative Prompt: {negative_prompt}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"True CFG Scale: {true_cfg_scale}")
    print(f"随机种子: {seed}")
    print(f"测试步数: {test_steps}")
    print("=" * 100)
    
    results = []
    
    for num_steps in test_steps:
        print(f"\n{'=' * 100}")
        print(f"测试 num_inference_steps={num_steps}")
        print(f"{'=' * 100}")
        
        try:
            # 生成 cfg_parallel_size=1 的图像
            print(f"\n[1/2] 生成图像 (cfg_parallel_size=1)...")
            img_without_cfg = generate_and_extract_image(
                model_name,
                prompt,
                negative_prompt,
                seed,
                height,
                width,
                num_steps,
                guidance_scale,
                true_cfg_scale,
                cfg_parallel_size=1
            )
            print(f"  ✓ 完成，形状: {img_without_cfg.shape}")
            
            # 生成 cfg_parallel_size=2 的图像
            print(f"\n[2/2] 生成图像 (cfg_parallel_size=2)...")
            img_with_cfg = generate_and_extract_image(
                model_name,
                prompt,
                negative_prompt,
                seed,
                height,
                width,
                num_steps,
                guidance_scale,
                true_cfg_scale,
                cfg_parallel_size=2
            )
            print(f"  ✓ 完成，形状: {img_with_cfg.shape}")
            
            # 对比差异
            max_diff, mean_diff, num_different, total_pixels = compare_outputs(
                img_with_cfg, img_without_cfg, 
                "cfg_parallel_size=2", "cfg_parallel_size=1"
            )
            
            results.append({
                'num_steps': num_steps,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'num_different': num_different,
                'total_pixels': total_pixels,
                'percent_different': num_different / total_pixels * 100,
            })
            
            # 保存图像
            output_dir = Path("diagnosis_outputs")
            output_dir.mkdir(exist_ok=True)
            from PIL import Image
            Image.fromarray(img_with_cfg).save(output_dir / f"steps{num_steps}_cfg_parallel.png")
            Image.fromarray(img_without_cfg).save(output_dir / f"steps{num_steps}_sequential.png")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'num_steps': num_steps,
                'error': str(e),
            })
    
    # 输出汇总报告
    print("\n" + "=" * 100)
    print("汇总报告")
    print("=" * 100)
    print(f"{'步数':<10} {'最大差异':<15} {'平均差异':<15} {'不同像素%':<15} {'状态'}")
    print("-" * 100)
    
    for r in results:
        if 'error' in r:
            print(f"{r['num_steps']:<10} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15} ❌")
        else:
            status = "✅ 无损" if r['max_diff'] == 0 else "❌ 有损"
            print(f"{r['num_steps']:<10} {r['max_diff']:<15.2f} {r['mean_diff']:<15.6f} {r['percent_different']:<15.2f} {status}")
    
    print("=" * 100)
    
    # 分析趋势
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        print("\n趋势分析:")
        lossless_steps = [r['num_steps'] for r in valid_results if r['max_diff'] == 0]
        lossy_steps = [r['num_steps'] for r in valid_results if r['max_diff'] > 0]
        
        if lossless_steps:
            print(f"  ✅ 无损的步数: {lossless_steps}")
        if lossy_steps:
            print(f"  ❌ 有损的步数: {lossy_steps}")
            print(f"\n  可能的原因:")
            print(f"    1. Scheduler 的 multistep 历史状态在 cfg-parallel 的两个 ranks 间不同步")
            print(f"    2. Scheduler 在步数 <= {max(lossless_steps) if lossless_steps else 0} 时处于 warmup 阶段")
            print(f"    3. 从步数 > {max(lossless_steps) if lossless_steps else 0} 开始使用完整的 multistep 求解器")
            print(f"    4. Rank 1 的 scheduler 内部状态未被更新，导致数值累积误差")


if __name__ == "__main__":
    diagnose_cfg_parallel()
