"""
测试脚本：对比开启和不开启 cfg-parallel 时的输出差异
验证 cfg-parallel 是否为无损加速（pixel 差异应该为零）
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"


def test_cfg_parallel_lossless():
    """
    对比开启和不开启 cfg-parallel 时的输出，验证是否无损。
    """
    model_name = "riverclouds/qwen_image_random"
    prompt = "a photo of a cat sitting on a laptop keyboard"
    seed = 42
    height = 256
    width = 256
    num_inference_steps = 4
    guidance_scale = 3.0  # 必须 > 1.0 才能触发 CFG
    
    m_with_cfg = None
    m_without_cfg = None
    
    try:
        print("=" * 60)
        print("测试模型:", model_name)
        print("提示词:", prompt)
        print("分辨率:", f"{width}x{height}")
        print("推理步数:", num_inference_steps)
        print("Guidance Scale:", guidance_scale)
        print("随机种子:", seed)
        print("=" * 60)
        
        # 测试开启 cfg_parallel
        print("\n[1/2] 测试开启 cfg_parallel...")
        m_with_cfg = Omni(
            model=model_name,
            enable_cfg_parallel=True,
        )
        
        outputs_with_cfg = m_with_cfg.generate(
            prompt,
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                num_outputs_per_prompt=1,
            ),
        )
        
        # 提取图像
        first_output_with_cfg = outputs_with_cfg[0]
        assert first_output_with_cfg.final_output_type == "image"
        req_out_with_cfg = first_output_with_cfg.request_output[0]
        assert isinstance(req_out_with_cfg, OmniRequestOutput)
        images_with_cfg = req_out_with_cfg.images
        assert len(images_with_cfg) == 1
        image_with_cfg = images_with_cfg[0]
        
        # 转换为 numpy 数组
        img_array_with_cfg = np.array(image_with_cfg)
        print(f"  ✓ 生成完成，图像形状: {img_array_with_cfg.shape}")
        
        # 关闭第一个模型以释放显存
        m_with_cfg.close()
        m_with_cfg = None
        
        # 测试不开启 cfg_parallel
        print("\n[2/2] 测试不开启 cfg_parallel...")
        m_without_cfg = Omni(
            model=model_name,
            enable_cfg_parallel=False,
        )
        
        outputs_without_cfg = m_without_cfg.generate(
            prompt,
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                num_outputs_per_prompt=1,
            ),
        )
        
        # 提取图像
        first_output_without_cfg = outputs_without_cfg[0]
        assert first_output_without_cfg.final_output_type == "image"
        req_out_without_cfg = first_output_without_cfg.request_output[0]
        assert isinstance(req_out_without_cfg, OmniRequestOutput)
        images_without_cfg = req_out_without_cfg.images
        assert len(images_without_cfg) == 1
        image_without_cfg = images_without_cfg[0]
        
        # 转换为 numpy 数组
        img_array_without_cfg = np.array(image_without_cfg)
        print(f"  ✓ 生成完成，图像形状: {img_array_without_cfg.shape}")
        
        # 验证形状一致
        print("\n" + "=" * 60)
        print("对比结果分析")
        print("=" * 60)
        
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
        
        # 保存图像供视觉检查
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        img_with_path = output_dir / "cfg_parallel_enabled.png"
        img_without_path = output_dir / "cfg_parallel_disabled.png"
        diff_path = output_dir / "pixel_difference.png"
        
        image_with_cfg.save(img_with_path)
        image_without_cfg.save(img_without_path)
        print(f"\n✓ 已保存图像:")
        print(f"  - cfg_parallel=True:  {img_with_path}")
        print(f"  - cfg_parallel=False: {img_without_path}")
        
        # 如果有差异，保存差异图
        if max_diff > 0:
            from PIL import Image
            # 将差异图归一化到 0-255 范围
            diff_normalized = (pixel_diff / max_diff * 255).astype(np.uint8)
            if len(diff_normalized.shape) == 3 and diff_normalized.shape[2] == 3:
                # RGB 图像，转换为灰度图以便查看
                diff_gray = np.mean(diff_normalized, axis=2).astype(np.uint8)
                diff_img = Image.fromarray(diff_gray, mode='L')
            else:
                diff_img = Image.fromarray(diff_normalized)
            diff_img.save(diff_path)
            print(f"  - 差异可视化:        {diff_path}")
        
        # 最终断言：cfg-parallel 应该是无损的
        print("\n" + "=" * 60)
        if max_diff == 0:
            print("✅ 测试通过: cfg-parallel 是无损加速，像素差异为零！")
            print("=" * 60)
        else:
            print("❌ 测试失败: cfg-parallel 存在像素差异！")
            print("=" * 60)
            raise AssertionError(
                f"cfg-parallel 应该是无损加速，但发现最大像素差异为 {max_diff}"
            )
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        raise
    finally:
        if m_with_cfg is not None and hasattr(m_with_cfg, "close"):
            m_with_cfg.close()
        if m_without_cfg is not None and hasattr(m_without_cfg, "close"):
            m_without_cfg.close()


if __name__ == "__main__":
    test_cfg_parallel_lossless()
