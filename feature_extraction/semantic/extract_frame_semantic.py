#!/usr/bin/env python3
"""
帧级别语义特征提取
为每一帧视频提取对应的语义特征向量，实现与视觉特征的时间对齐
"""

import os
import numpy as np
import torch
import glob
import json
import warnings
from tqdm import tqdm
import argparse
from pathlib import Path

# 导入现有的语义特征提取模块
from .extract_whisperx import transcribe_with_whisperx
from .extract_mclip import extract_mclip_embeddings
from .extract_xlmr import extract_xlmr_embeddings


def get_video_frame_count(frame_dir, video_id):
    """
    获取视频的帧数
    
    Args:
        frame_dir: 帧目录根路径
        video_id: 视频ID
        
    Returns:
        int: 帧数，如果无法确定则返回0
    """
    # 构建视频帧目录路径
    video_frame_dir = os.path.join(frame_dir, video_id)
    
    if not os.path.exists(video_frame_dir):
        print(f"⚠️ Frame directory not found: {video_frame_dir}")
        return 0
    
    # 支持常见图像格式
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    frame_files = []
    
    for pattern in patterns:
        frame_files.extend(glob.glob(os.path.join(video_frame_dir, pattern)))
    
    frame_count = len(frame_files)
    print(f"📊 Found {frame_count} frames for video {video_id}")
    return frame_count


def align_words_to_frames(word_segments, frame_count, fps=25.0):
    """
    将词级别的时间戳对齐到视频帧
    
    Args:
        word_segments: WhisperX输出的词级别片段
        frame_count: 视频总帧数
        fps: 视频帧率
        
    Returns:
        list: 每帧对应的词列表
    """
    # 初始化每帧的词列表
    frame_words = [[] for _ in range(frame_count)]
    
    # 计算视频总时长
    video_duration = frame_count / fps
    
    for segment in word_segments.get("segments", []):
        words = segment.get("words", [])
        
        for word_info in words:
            if "start" in word_info and "end" in word_info:
                start_time = word_info["start"]
                end_time = word_info["end"]
                word_text = word_info.get("word", "").strip()
                
                if not word_text:
                    continue
                
                # 计算词对应的帧范围
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                # 确保帧索引在有效范围内
                start_frame = max(0, min(start_frame, frame_count - 1))
                end_frame = max(0, min(end_frame, frame_count - 1))
                
                # 将词分配给对应的帧
                for frame_idx in range(start_frame, end_frame + 1):
                    if frame_idx < frame_count:
                        frame_words[frame_idx].append({
                            'word': word_text,
                            'start': start_time,
                            'end': end_time,
                            'confidence': word_info.get('score', 1.0)
                        })
    
    return frame_words


def extract_text_embeddings(texts, model_type='mclip', model_name=None, device='cuda'):
    """
    提取文本嵌入向量
    
    Args:
        texts: 文本列表
        model_type: 模型类型 ('mclip' 或 'xlmr')
        model_name: 模型名称
        device: 计算设备
        
    Returns:
        numpy.ndarray: 嵌入向量数组
    """
    if not texts:
        # 返回零向量
        if model_type == 'mclip':
            return np.zeros((1, 512), dtype=np.float32)  # M-CLIP通常是512维
        else:  # xlmr
            return np.zeros((1, 768), dtype=np.float32)  # XLM-R base是768维
    
    # 创建临时文件存储文本
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
        temp_txt_path = f.name
    
    try:
        if model_type == 'mclip':
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                temp_npy_path = f.name
            
            # 提取M-CLIP嵌入
            extract_mclip_embeddings(
                input_txt=temp_txt_path,
                output_npy=temp_npy_path,
                model_name=model_name or 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
                batch_size=16,
                device=device
            )
            
            # 加载结果
            embeddings = np.load(temp_npy_path)
            os.unlink(temp_npy_path)
            
        else:  # xlmr
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                temp_npy_path = f.name
            
            # 提取XLM-R嵌入
            extract_xlmr_embeddings(
                input_txt=temp_txt_path,
                output_npy=temp_npy_path,
                model_name=model_name or 'xlm-roberta-base',
                device=device
            )
            
            # 加载结果
            embeddings = np.load(temp_npy_path)
            os.unlink(temp_npy_path)
        
        # 清理临时文件
        os.unlink(temp_txt_path)
        
        return embeddings
        
    except Exception as e:
        print(f"⚠️ Error extracting embeddings: {e}")
        # 清理临时文件
        try:
            os.unlink(temp_txt_path)
        except:
            pass
        
        # 返回零向量
        if model_type == 'mclip':
            return np.zeros((len(texts), 512), dtype=np.float32)
        else:
            return np.zeros((len(texts), 768), dtype=np.float32)


def semantic_per_frame(
    audio_path,
    frame_count,
    fps=25.0,
    output_dir=None,
    device='cuda',
    whisper_model='large-v2',
    mclip_model='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
    xlmr_model='xlm-roberta-base',
    use_both_models=True
):
    """
    为每帧提取语义特征并保存
    
    Args:
        audio_path: 音频文件路径
        frame_count: 视频总帧数
        fps: 视频帧率
        output_dir: 输出目录
        device: 计算设备
        whisper_model: Whisper模型名称
        mclip_model: M-CLIP模型名称
        xlmr_model: XLM-R模型名称
        use_both_models: 是否同时使用两个模型
        
    Returns:
        bool: 是否成功提取
    """
    if output_dir is None:
        video_id = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join("semantic", video_id)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print(f"🎬 Processing {frame_count} frames at {fps} FPS")
    
    try:
        # 步骤1: 使用WhisperX进行转录和对齐
        print("🎙️ Step 1: Transcribing audio with WhisperX...")
        
        # 创建临时目录用于WhisperX输出
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # 根据设备选择合适的计算类型和批处理大小（使用保守设置防止bus error）
            compute_type = "float16" if device == "cuda" else "float32"
            
            # 使用更保守的批处理大小以避免内存问题
            import whisperx
            try:
                # 检查音频时长以调整批处理大小
                audio_for_check = whisperx.load_audio(audio_path)
                duration = len(audio_for_check) / 16000
                
                if duration < 30:
                    batch_size = 1  # 短音频使用最小批处理
                elif duration < 120:
                    batch_size = 4 if device == "cuda" else 2
                else:
                    batch_size = 8 if device == "cuda" else 4
                    
                print(f"🎵 Audio duration: {duration:.2f}s, using batch_size: {batch_size}")
                
            except Exception as e:
                print(f"⚠️ Could not determine audio duration, using conservative batch size: {e}")
                batch_size = 4 if device == "cuda" else 2
            
            whisperx_result = transcribe_with_whisperx(
                audio_path=audio_path,
                output_dir=temp_dir,
                model_name=whisper_model,
                device=device,
                batch_size=batch_size,
                compute_type=compute_type
            )
            
            if not whisperx_result:
                print("❌ WhisperX transcription failed")
                return False
            
            # 步骤2: 将词对齐到帧
            print("🔗 Step 2: Aligning words to frames...")
            frame_words = align_words_to_frames(whisperx_result, frame_count, fps)
            
            # 步骤3: 为每帧生成文本并提取特征
            print("🧠 Step 3: Extracting semantic features per frame...")
            
            saved_count = 0
            
            # 分批处理以节省内存，GPU可以处理更大的批处理
            processing_batch_size = 64 if device == "cuda" else 32
            for batch_start in tqdm(range(0, frame_count, processing_batch_size), desc="Processing frames"):
                batch_end = min(batch_start + processing_batch_size, frame_count)
                batch_texts = []
                batch_indices = []
                
                # 准备批处理文本
                for frame_idx in range(batch_start, batch_end):
                    words = frame_words[frame_idx]
                    if words:
                        # 合并当前帧的所有词
                        frame_text = ' '.join([w['word'] for w in words])
                        batch_texts.append(frame_text.strip())
                    else:
                        # 使用空字符串或前一帧的文本
                        if frame_idx > 0 and batch_texts:
                            batch_texts.append(batch_texts[-1])  # 使用前一帧
                        else:
                            batch_texts.append("")  # 空文本
                    
                    batch_indices.append(frame_idx)
                
                # 提取嵌入向量
                if use_both_models:
                    # M-CLIP嵌入
                    try:
                        mclip_embeddings = extract_text_embeddings(
                            batch_texts, 'mclip', mclip_model, device
                        )
                        
                        # 保存M-CLIP特征
                        for i, frame_idx in enumerate(batch_indices):
                            out_path = os.path.join(output_dir, f"mclip_{frame_idx:05d}.npy")
                            try:
                                if i < len(mclip_embeddings):
                                    np.save(out_path, mclip_embeddings[i].astype(np.float32))
                                else:
                                    # 使用零向量
                                    np.save(out_path, np.zeros(512, dtype=np.float32))
                                saved_count += 1
                            except Exception as e:
                                print(f"⚠️ Failed to save M-CLIP feature for frame {frame_idx}: {e}")
                    except Exception as e:
                        print(f"⚠️ M-CLIP extraction failed for batch: {e}")
                    
                    # XLM-R嵌入
                    try:
                        xlmr_embeddings = extract_text_embeddings(
                            batch_texts, 'xlmr', xlmr_model, device
                        )
                        
                        # 保存XLM-R特征
                        for i, frame_idx in enumerate(batch_indices):
                            out_path = os.path.join(output_dir, f"xlmr_{frame_idx:05d}.npy")
                            try:
                                if i < len(xlmr_embeddings):
                                    np.save(out_path, xlmr_embeddings[i].astype(np.float32))
                                else:
                                    # 使用零向量
                                    np.save(out_path, np.zeros(768, dtype=np.float32))
                            except Exception as e:
                                print(f"⚠️ Failed to save XLM-R feature for frame {frame_idx}: {e}")
                    except Exception as e:
                        print(f"⚠️ XLM-R extraction failed for batch: {e}")
                
                else:
                    # 只使用M-CLIP
                    try:
                        mclip_embeddings = extract_text_embeddings(
                            batch_texts, 'mclip', mclip_model, device
                        )
                        
                        # 保存特征
                        for i, frame_idx in enumerate(batch_indices):
                            out_path = os.path.join(output_dir, f"semantic_{frame_idx:05d}.npy")
                            try:
                                if i < len(mclip_embeddings):
                                    np.save(out_path, mclip_embeddings[i].astype(np.float32))
                                else:
                                    np.save(out_path, np.zeros(512, dtype=np.float32))
                                saved_count += 1
                            except Exception as e:
                                print(f"⚠️ Failed to save feature for frame {frame_idx}: {e}")
                    except Exception as e:
                        print(f"⚠️ Feature extraction failed for batch: {e}")
            
            # 保存元数据
            meta = {
                'audio_path': audio_path,
                'frame_count': frame_count,
                'fps': fps,
                'whisper_model': whisper_model,
                'mclip_model': mclip_model,
                'xlmr_model': xlmr_model,
                'use_both_models': use_both_models,
                'saved_features': saved_count,
                'total_words': sum(len(words) for words in frame_words),
                'frames_with_speech': sum(1 for words in frame_words if words)
            }
            
            meta_path = os.path.join(output_dir, 'semantic_meta.json')
            try:
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                print(f"📋 Metadata saved to {meta_path}")
            except Exception as e:
                print(f"⚠️ Failed to save metadata: {e}")
            
            print(f"✅ Saved semantic features for {frame_count} frames to {output_dir}")
            print(f"📊 Statistics: {meta['frames_with_speech']}/{frame_count} frames have speech")
            
            return saved_count > 0
    
    except Exception as e:
        print(f"❌ Error during semantic feature extraction: {e}")
        return False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Extract per-frame semantic features")
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file')
    parser.add_argument('--frame_count', type=int, required=True, help='Total number of video frames')
    parser.add_argument('--fps', type=float, default=25.0, help='Video frame rate (default: 25.0)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-generated if not provided)')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--whisper_model', type=str, default='large-v2', help='Whisper model name')
    parser.add_argument('--mclip_model', type=str, default='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus', help='M-CLIP model name')
    parser.add_argument('--xlmr_model', type=str, default='xlm-roberta-base', help='XLM-R model name')
    parser.add_argument('--mclip_only', action='store_true', help='Use only M-CLIP model')
    
    args = parser.parse_args()
    
    success = semantic_per_frame(
        audio_path=args.audio_path,
        frame_count=args.frame_count,
        fps=args.fps,
        output_dir=args.output_dir,
        device=args.device,
        whisper_model=args.whisper_model,
        mclip_model=args.mclip_model,
        xlmr_model=args.xlmr_model,
        use_both_models=not args.mclip_only
    )
    
    if success:
        print("🎉 Semantic frame feature extraction completed successfully!")
    else:
        print("❌ Semantic frame feature extraction failed!")
        exit(1)


if __name__ == "__main__":
    main()