import argparse
import torch
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm

from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2
import torchaudio
import scipy.io.wavfile as sciowav
import scipy.signal as scisig
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Add debug flags
DEBUG_AUDIO = True
DEBUG_MODEL = True

parser = argparse.ArgumentParser()
parser.add_argument("--img_root", type=str, default="/root/autodl-tmp/eval/frames")
parser.add_argument("--json_path", type=str, default="/root/autodl-tmp/eval/annotations.json")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitb14")
parser.add_argument("--ckpt_path", type=str, default="/root/gazelle/scripts/experiments/train_gazefollow_audio_debug/2025-09-01_09-01-10/best_model.pth")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--use_audio_model", action='store_true', help="construct model with audio branch enabled")
parser.add_argument("--audio_roots", type=str, nargs='+', default=None, help="audio feature root dirs (optional)")
parser.add_argument("--audio_dims", type=int, nargs='+', default=None, help="audio feature dims (must match roots)")
parser.add_argument("--zero_audio_alpha", action='store_true', help="zero out audio_alpha params at eval to simulate no-audio")
parser.add_argument("--audio_wav_root", type=str, default=None, help="root dir containing <video_id>.wav files for on-the-fly feature extraction")
parser.add_argument("--audio_fps", type=int, default=30, help="assumed frame rate for frame-to-time mapping")
parser.add_argument("--audio_win_sec", type=float, default=5, help="audio window length (seconds) centered on frame time")
args = parser.parse_args()


def parse_frame_and_pid(name: str):
    """
    '0041000.jpg' -> ('004001.jpg', pid=1)
    """
    base = name.split('.')[0]   
    video_id = base[:3]        
    pid = int(base[3])         
    frame_num = int(base[4:]) + 1 
    frame = f"{video_id}{frame_num:03d}" 
    return f"{frame}.jpg", pid


class VGSDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_root, transform, audio_wav_root=None, audio_fps=30, audio_win_sec=0.5):
        self.items = []
        self.img_root = img_root
        self.transform = transform
        self.audio_wav_root = audio_wav_root
        self.audio_fps = audio_fps
        self.audio_win_sec = audio_win_sec
        self.video_to_wav = {}

        if not os.path.isfile(json_path):
            raise RuntimeError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        for item in data:
            fname = item.get("fname") or item.get("file_name") or item.get("path")
            if fname is None:
                continue
            frame_name, _ = parse_frame_and_pid(os.path.basename(fname))
            img_path = os.path.join(img_root, frame_name)
            if not os.path.isfile(img_path):
                continue

            if "bbox" in item:
                x1, y1, x2, y2 = item["bbox"]
            else:
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]

            if "gaze" in item:
                gx, gy = item["gaze"][:2]
            else:
                gx, gy = item["gx"], item["gy"]

            vid = frame_name[:3]
            frame_idx = int(frame_name[3:6]) - 1
            self.items.append((img_path, [x1, y1, x2, y2], [gx, gy], vid, frame_idx))

    def __getitem__(self, idx):
        img_path, bbox, gaze, video_id, frame_idx = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        h, w = image.size[1], image.size[0]
        image_t = self.transform(image)

        # 归一化
        bbox_norm = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
        gazex_norm = gaze[0] / w
        gazey_norm = gaze[1] / h

        sample = {
            "image": image_t,
            "bboxes": [bbox_norm],
            "gazex": [gazex_norm],
            "gazey": [gazey_norm],
            "height": h,
            "width": w,
            "video_id": video_id,
            "frame_idx": frame_idx,
        }
        return sample

    def __len__(self):
        return len(self.items)


def collate(batch):
    images = torch.stack([b["image"] for b in batch])
    bboxes = [b["bboxes"] for b in batch]
    gazex = [b["gazex"] for b in batch]
    gazey = [b["gazey"] for b in batch]
    height = [b["height"] for b in batch]
    width = [b["width"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    frame_idxs = [b["frame_idx"] for b in batch]
    return images, bboxes, gazex, gazey, height, width, video_ids, frame_idxs


def debug_model_structure(model):
    """Debug function to inspect model structure"""
    print("\n========== MODEL DEBUG INFO ==========")
    print(f"Model type: {type(model)}")
    
    # Check if model has audio components
    has_audio_alpha = hasattr(model, 'audio_alpha')
    has_audio_proj = hasattr(model, 'audio_proj') or hasattr(model, 'audio_projection')
    has_audio_encoder = hasattr(model, 'audio_encoder')
    
    print(f"Has audio_alpha: {has_audio_alpha}")
    print(f"Has audio_proj: {has_audio_proj}")
    print(f"Has audio_encoder: {has_audio_encoder}")
    
    if has_audio_alpha:
        print(f"audio_alpha type: {type(model.audio_alpha)}")
        if isinstance(model.audio_alpha, torch.nn.Parameter):
            print(f"audio_alpha value: {model.audio_alpha.data}")
        else:
            print(f"audio_alpha parameters: {list(model.audio_alpha.parameters())}")
    
    # Print model architecture summary
    print(f"\nModel parameters:")
    for name, param in model.named_parameters():
        if 'audio' in name.lower():
            print(f"  {name}: {param.shape}, requires_grad: {param.requires_grad}")
    
    return has_audio_alpha, has_audio_proj, has_audio_encoder


def debug_audio_features(audio_features, batch_idx=0):
    """Debug function to inspect audio features"""
    if audio_features is None:
        print(f"Batch {batch_idx}: No audio features")
        return
    
    print(f"\nBatch {batch_idx} Audio Debug:")
    print(f"  Shape: {audio_features.shape}")
    print(f"  Mean: {audio_features.mean().item():.6f}")
    print(f"  Std: {audio_features.std().item():.6f}")
    print(f"  Min: {audio_features.min().item():.6f}")
    print(f"  Max: {audio_features.max().item():.6f}")
    print(f"  Zeros ratio: {(audio_features == 0).float().mean().item():.4f}")


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # Debug: Print configuration
    print("\n========== CONFIGURATION DEBUG ==========")
    print(f"use_audio_model: {args.use_audio_model}")
    print(f"audio_wav_root: {args.audio_wav_root}")
    print(f"zero_audio_alpha: {args.zero_audio_alpha}")
    print(f"audio_roots: {args.audio_roots}")
    print(f"audio_dims: {args.audio_dims}")

    total_audio_dim = None
    if args.use_audio_model:
        if args.audio_roots is not None and args.audio_dims is not None:
            assert len(args.audio_roots) == len(args.audio_dims)
            total_audio_dim = int(sum(args.audio_dims))
    
    print(f"total_audio_dim: {total_audio_dim}")
    
    if args.use_audio_model and (total_audio_dim is not None):
        model, transform = get_gazelle_model(args.model_name, use_audio=True, audio_dim=total_audio_dim)
    else:
        # fall back to model defaults (audio_dim=856) if not provided
        model, transform = get_gazelle_model(args.model_name, use_audio=args.use_audio_model)
    
    # Debug model structure before loading checkpoint
    if DEBUG_MODEL:
        debug_model_structure(model)
    
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    # Debug: Check audio_alpha after loading checkpoint
    if args.zero_audio_alpha and hasattr(model, 'audio_alpha'):
        print(f"\nBefore zeroing audio_alpha: {model.audio_alpha.data if isinstance(model.audio_alpha, torch.nn.Parameter) else 'Not a Parameter'}")
        with torch.no_grad():
            try:
                if isinstance(model.audio_alpha, torch.nn.Parameter):
                    model.audio_alpha.zero_()
                else:
                    for p in model.audio_alpha.parameters():
                        p.zero_()
                print("Zeroed model.audio_alpha for no-audio simulation.")
                print(f"After zeroing audio_alpha: {model.audio_alpha.data if isinstance(model.audio_alpha, torch.nn.Parameter) else 'Not a Parameter'}")
            except Exception as e:
                print(f"Error zeroing audio_alpha: {e}")

    dataset = VGSDataset(args.json_path, args.img_root, transform,
                         audio_wav_root=args.audio_wav_root,
                         audio_fps=args.audio_fps,
                         audio_win_sec=args.audio_win_sec)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    # Prepare audio model if requested
    wav2vec_model = None
    wav2vec_processor = None
    if args.use_audio_model and (args.audio_wav_root is not None):
        try:
            wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
            wav2vec_model.eval()
            print("Wav2Vec2 model loaded successfully")
        except Exception as e:
            print(f"Wav2Vec2 not available (offline). Falling back to MFCC-only features. Err: {e}")
            wav2vec_model = None
            wav2vec_processor = None

    # Prepare MFCC extractor (CPU) with no center padding to avoid short-segment errors
    sr_target = 16000
    mfcc_transform = None
    if args.use_audio_model and (args.audio_wav_root is not None):
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr_target,
            n_mfcc=29,
            melkwargs={
                "n_mels": 40,
                "n_fft": 400,
                "hop_length": 160,
                "center": False,
            },
        )

    aucs, min_l2s, avg_l2s = [], [], []
    total_samples = 0
    valid_samples = 0
    inside_bbox_samples = 0
    
    # Store predictions for comparison
    predictions_with_audio = []
    predictions_without_audio = []

    for batch_idx, (images, bboxes, gazex, gazey, height, width, video_ids, frame_idxs) in tqdm(
        enumerate(dataloader), desc="Evaluating", total=len(dataloader)
    ):
        input_dict = {"images": images.to(device), "bboxes": bboxes}
        
        # Audio feature extraction
        audio_included = False
        if args.use_audio_model and (args.audio_wav_root is not None):
            # Build per-image audio features by slicing wav around frame time
            batch_audio_features = []
            half_win = int(args.audio_win_sec * sr_target / 2)
            
            for i, (vid, fidx) in enumerate(zip(video_ids, frame_idxs)):
                wav_path = os.path.join(args.audio_wav_root, f"{vid}.wav")
                if not os.path.isfile(wav_path):
                    if DEBUG_AUDIO and i == 0:
                        print(f"WAV file not found: {wav_path}")
                    batch_audio_features.append(torch.zeros(856, dtype=torch.float32))
                    continue
                
                try:
                    # Load WAV via SciPy to avoid torchaudio backend issues
                    sr, wav_np = sciowav.read(wav_path)
                    if wav_np.ndim > 1:
                        wav_np = wav_np.mean(axis=1)
                    # convert to float32 in [-1, 1]
                    if wav_np.dtype != 'float32':
                        max_val = np.iinfo(wav_np.dtype).max if np.issubdtype(wav_np.dtype, np.integer) else 1.0
                        wav_np = wav_np.astype(np.float32) / float(max_val)
                    if sr != sr_target:
                        # resample using polyphase filtering
                        gcd = np.gcd(sr, sr_target)
                        up = sr_target // gcd
                        down = sr // gcd
                        wav_np = scisig.resample_poly(wav_np, up, down).astype(np.float32)
                    wav = torch.from_numpy(wav_np)
                    
                    # center time of frame (seconds)
                    t = (fidx / max(1, args.audio_fps))
                    center = int(t * sr_target)
                    start = max(0, center - half_win)
                    end = min(wav.numel(), center + half_win)
                    segment = wav[start:end]
                    
                    if segment.numel() == 0:
                        if DEBUG_AUDIO and i == 0:
                            print(f"Empty audio segment for {vid}, frame {fidx}")
                        batch_audio_features.append(torch.zeros(856, dtype=torch.float32))
                        continue

                    # 1) compute lightweight 88-d MFCC-based descriptor
                    if segment.numel() < 400:
                        segment = torch.nn.functional.pad(segment, (0, 400 - segment.numel()))
                    mfcc = mfcc_transform(segment.unsqueeze(0))  # (1, 29, T)
                    mfcc = mfcc.squeeze(0)
                    delta = torchaudio.functional.compute_deltas(mfcc)
                    delta2 = torchaudio.functional.compute_deltas(delta)
                    mfcc_mean = mfcc.mean(dim=1)
                    delta_mean = delta.mean(dim=1)
                    delta2_mean = delta2.mean(dim=1)
                    rms = torch.sqrt((segment.float() ** 2).mean()).unsqueeze(0)
                    feat_88 = torch.cat([rms, mfcc_mean, delta_mean, delta2_mean], dim=0)  # (88,)

                    # 2) wav2vec 768, or zeros if unavailable
                    if (wav2vec_model is not None) and (wav2vec_processor is not None):
                        inputs = wav2vec_processor(segment.unsqueeze(0), sampling_rate=sr_target, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = wav2vec_model(**inputs)
                        feat_768 = outputs.last_hidden_state.mean(dim=1).squeeze(0).float().detach().cpu()
                    else:
                        feat_768 = torch.zeros(768, dtype=torch.float32)

                    feat = torch.cat([feat_88, feat_768], dim=0)  # (856,)
                    batch_audio_features.append(feat)
                    audio_included = True
                    
                except Exception as e:
                    if DEBUG_AUDIO and i == 0:
                        print(f"Error processing audio for {vid}: {e}")
                    batch_audio_features.append(torch.zeros(856, dtype=torch.float32))
            
            audio_tensor = torch.stack(batch_audio_features).to(device)
            input_dict["audio"] = audio_tensor
            
            # Debug audio features for first few batches
            if DEBUG_AUDIO and batch_idx < 3:
                debug_audio_features(audio_tensor, batch_idx)
                print(f"  Audio included: {audio_included}")

        # Get model predictions
        preds = model.forward(input_dict)
        
        # Debug: Store some predictions for comparison
        if batch_idx < 3:
            if "audio" in input_dict:
                predictions_with_audio.append(preds['heatmap'][0][0].detach().cpu().numpy() if len(preds['heatmap']) > 0 else None)
            else:
                predictions_without_audio.append(preds['heatmap'][0][0].detach().cpu().numpy() if len(preds['heatmap']) > 0 else None)

        for i in range(images.shape[0]):   # 每张图
            for j in range(len(bboxes[i])):  # 每个 head
                total_samples += 1
                
                # 检查注视点是否在bbox外
                gx, gy = gazex[i][j], gazey[i][j]
                if gx < 0 or gy < 0:  # 无效注视点
                    continue
                
                x1, y1, x2, y2 = bboxes[i][j]
                
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    # 注视点在bbox内，跳过这个样本
                    inside_bbox_samples += 1
                    continue
                
                valid_samples += 1
                gx_list = [gx]
                gy_list = [gy]
                auc = gazefollow_auc(preds['heatmap'][i][j], gx_list, gy_list, height[i], width[i])
                avg_l2, min_l2 = gazefollow_l2(preds['heatmap'][i][j], gx_list, gy_list)
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

    # Debug: Compare predictions if we have both
    if len(predictions_with_audio) > 0 and len(predictions_without_audio) > 0:
        print("\n========== PREDICTION COMPARISON ==========")
        for i in range(min(len(predictions_with_audio), len(predictions_without_audio))):
            if predictions_with_audio[i] is not None and predictions_without_audio[i] is not None:
                diff = np.abs(predictions_with_audio[i] - predictions_without_audio[i]).mean()
                print(f"Batch {i} - Mean absolute difference: {diff:.8f}")

    print("========== Sample Statistics ==========")
    print(f"Total samples: {total_samples}")
    print(f"Samples with gaze inside bbox: {inside_bbox_samples}")
    print(f"Valid samples (gaze outside bbox): {valid_samples}")
    if total_samples > 0:
        print(f"Percentage of valid samples: {valid_samples/total_samples*100:.1f}%")
    else:
        print("Percentage of valid samples: N/A (no samples)")
    
    print("\n========== Results (Gaze Outside BBox Only) ==========")
    if len(aucs) > 0:
        print("AUC: {:.4f}".format(np.array(aucs).mean()))
        print("Avg L2: {:.4f}".format(np.array(avg_l2s).mean()))
        print("Min L2: {:.4f}".format(np.array(min_l2s).mean()))
    else:
        print("No valid samples found!")


if __name__ == "__main__":
    main()