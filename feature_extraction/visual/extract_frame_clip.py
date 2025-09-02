#!/usr/bin/env python3
"""
帧级别CLIP视觉特征提取
为每一帧视频图像提取CLIP特征向量
"""

import os
import numpy as np
import torch
import glob
from PIL import Image
from tqdm import tqdm
import argparse
import json
import warnings


def load_clip_model(model_name='ViT-B-16-plus-240', pretrained='laion400m_e32', device='cuda'):
    """
    加载CLIP模型
    
    Args:
        model_name: CLIP模型名称
        pretrained: 预训练权重
        device: 计算设备
    
    Returns:
        tuple: (model, preprocess, device)
    """
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        print(f"✅ CLIP model '{model_name}' loaded on {device}")
        return model, preprocess, device
    except Exception as e:
        warnings.warn(f"⚠️ Failed to load CLIP model: {e}")
        return None, None, device


def extract_clip_from_frame(model, preprocess, image_path, device):
    """
    从单帧图像提取CLIP特征
    
    Args:
        model: CLIP模型
        preprocess: 图像预处理函数
        image_path: 图像文件路径
        device: 计算设备
    
    Returns:
        numpy.ndarray: CLIP特征向量
    """
    try:
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # 归一化
            
        return features.cpu().numpy().squeeze()
    except Exception as e:
        warnings.warn(f"⚠️ Failed to extract CLIP features from {image_path}: {e}")
        return None


def get_frame_files(frame_dir):
    """
    获取帧目录中的所有图像文件，按帧号排序
    
    Args:
        frame_dir: 视频帧目录
    
    Returns:
        list: 排序后的图像文件路径列表
    """
    # 支持常见图像格式
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(frame_dir, pattern)))
    
    # 按文件名数字顺序排序
    def extract_frame_number(filepath):
        filename = os.path.basename(filepath)
        # 尝试提取文件名中的数字作为帧号
        try:
            numbers = ''.join(filter(str.isdigit, filename))
            return int(numbers) if numbers else 0
        except:
            return 0
    
    image_files.sort(key=extract_frame_number)
    return image_files


def clip_per_frame(
    frame_dir,
    output_dir=None,
    model_name='ViT-B-16-plus-240',
    pretrained='laion400m_e32',
    device='cuda',
    batch_size=16
):
    """
    为每帧提取CLIP特征并保存
    
    Args:
        frame_dir: 视频帧目录
        output_dir: 输出目录 (如果为None，将自动创建clip/{video_id}/)
        model_name: CLIP模型名称
        pretrained: 预训练权重
        device: 计算设备
        batch_size: 批处理大小
    
    Returns:
        bool: 是否成功提取
    """
    # 如果没有指定输出目录，自动创建符合规范的目录结构
    if output_dir is None:
        video_id = os.path.basename(frame_dir.rstrip('/'))
        output_dir = os.path.join("clip", video_id)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # 获取所有帧文件
    frame_files = get_frame_files(frame_dir)
    if not frame_files:
        warnings.warn(f"⚠️ No image files found in {frame_dir}")
        return False
    
    print(f"🖼️ Found {len(frame_files)} frames in {frame_dir}")
    
    # 加载CLIP模型
    model, preprocess, device = load_clip_model(model_name, pretrained, device)
    if model is None:
        return False
    
    saved_count = 0
    
    # 分批处理以节省内存
    for i in tqdm(range(0, len(frame_files), batch_size), desc="Extracting CLIP features"):
        batch_files = frame_files[i:i+batch_size]
        batch_images = []
        batch_indices = []
        
        # 准备批处理数据
        for j, frame_path in enumerate(batch_files):
            try:
                image = Image.open(frame_path).convert('RGB')
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)
                batch_indices.append(i + j)
            except Exception as e:
                warnings.warn(f"⚠️ Failed to load {frame_path}: {e}")
                # 使用零向量作为占位符
                batch_indices.append(i + j)
                batch_images.append(None)
        
        if not batch_images:
            continue
        
        # 过滤None值
        valid_images = [img for img in batch_images if img is not None]
        valid_indices = [idx for idx, img in zip(batch_indices, batch_images) if img is not None]
        
        if valid_images:
            try:
                # 批量处理
                batch_tensor = torch.stack(valid_images).to(device)
                with torch.no_grad():
                    features = model.encode_image(batch_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features = features.cpu().numpy()
                
                # 保存每帧特征
                for idx, feat in zip(valid_indices, features):
                    out_path = os.path.join(output_dir, f"clip_{idx:05d}.npy")
                    try:
                        np.save(out_path, feat.astype(np.float32))
                        saved_count += 1
                    except Exception as e:
                        warnings.warn(f"⚠️ Failed to save {out_path}: {e}")
                        
            except Exception as e:
                warnings.warn(f"⚠️ Batch processing failed: {e}")
        
        # 处理失败的帧，保存零向量
        for idx, img in zip(batch_indices, batch_images):
            if img is None:
                out_path = os.path.join(output_dir, f"clip_{idx:05d}.npy")
                try:
                    # 使用默认CLIP特征维度（通常是512）
                    zero_feat = np.zeros(512, dtype=np.float32)
                    np.save(out_path, zero_feat)
                    saved_count += 1
                except Exception as e:
                    warnings.warn(f"⚠️ Failed to save zero vector {out_path}: {e}")
    
    print(f"✅ Saved {saved_count}/{len(frame_files)} CLIP features to {output_dir}")
    
    # 保存元数据
    meta = {
        'frame_dir': frame_dir,
        'frame_count': len(frame_files),
        'model_name': model_name,
        'pretrained': pretrained,
        'feature_dim': 512,  # 大多数CLIP模型使用512维
        'saved_features': saved_count
    }
    
    meta_path = os.path.join(output_dir, 'clip_meta.json')
    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"📋 Metadata saved to {meta_path}")
    except Exception as e:
        warnings.warn(f"⚠️ Failed to save metadata: {e}")
    
    return saved_count > 0


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Extract per-frame CLIP visual features")
    parser.add_argument('--frame_dir', type=str, required=True, help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for CLIP features (auto-generated if not provided)')
    parser.add_argument('--model_name', type=str, default='ViT-B-16-plus-240', help='CLIP model name')
    parser.add_argument('--pretrained', type=str, default='laion400m_e32', help='Pretrained weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    
    args = parser.parse_args()
    
    success = clip_per_frame(
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size
    )
    
    if success:
        print("🎉 CLIP feature extraction completed successfully!")
    else:
        print("❌ CLIP feature extraction failed!")
        exit(1)


if __name__ == "__main__":
    main()