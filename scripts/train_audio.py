import argparse
from datetime import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14")
parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/.autodl/train/processed/frames')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='train_gazefollow_audio_debug')
parser.add_argument('--log_iter', type=int, default=10)
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=120)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--image_root', type=str, default=None)

# audio
parser.add_argument('--use_audio', action='store_true')
parser.add_argument('--audio_roots', type=str, nargs='+', default=None)
parser.add_argument('--audio_dims', type=int, nargs='+', default=None)
parser.add_argument('--reg_lambda', type=float, default=1e-5)
parser.add_argument('--strict_load', action='store_true', help='Abort if checkpoint keys mismatch when set')

# 视觉-only ckpt（来自数据集 A）
parser.add_argument('--resume_ckpt', type=str, default=None, help='Path to visual-only ckpt from dataset A')
args = parser.parse_args()


def _ensure_heatmap_BHW(h):
    """将模型返回的 heatmap 统一成 [B,H,W]。"""
    if isinstance(h, torch.Tensor):
        if h.dim() == 4 and h.size(1) == 1:
            return h.squeeze(1)
        if h.dim() == 3:
            return h
    elif isinstance(h, (list, tuple)):
        h = torch.stack(h, dim=0)
        return _ensure_heatmap_BHW(h)
    raise ValueError(f"Unsupported heatmap type/shape: {type(h)} | {getattr(h, 'shape', None)}")


def _ensure_bbox_list_of_lists(bboxes):
    """
    把 collate 出来的 bboxes 统一成 list[list[4]]：
    - 若元素本身是多框（list/tuple 且第0个还是 list/tuple/Tensor），原样返回；
    - 否则视为单框，包一层 -> [box]。
    """
    out = []
    for b in bboxes:
        if isinstance(b, (list, tuple)) and len(b) > 0 and isinstance(b[0], (list, tuple, torch.Tensor)):
            out.append(b)      # 已是多框
        else:
            out.append([b])    # 单框 -> 包一层
    return out


def main():
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir, exist_ok=True)

    total_audio_dim = sum(args.audio_dims) if args.use_audio and args.audio_dims else 0
    if args.use_audio and total_audio_dim == 0:
        raise ValueError('--use_audio requires valid --audio_dims')
    model, transform = get_gazelle_model(args.model, use_audio=args.use_audio, audio_dim=total_audio_dim)
    model.cuda()

    # 从 A 的视觉 ckpt 加载（strict=False 忽略音频分支）
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        print(f"\n🔄 Loading checkpoint from: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)  # 兼容两种保存格式
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"   missing keys: {missing}")
        print(f"   unexpected keys: {unexpected}")
        if args.strict_load and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError('Checkpoint incompatible with model')
        if 'linear.weight' in state:
            w = state['linear.weight']
            try:
                print(f"   loaded linear.weight mean: {w.float().mean().item():.6f}")
            except Exception:
                pass
    else:
        print("\nℹ️ No --resume_ckpt provided (or file not found); training without loading A weights.")

    # 冻结视觉骨干
    for param in model.backbone.parameters():
        param.requires_grad = False

    print("\n🔍 模型结构如下：")
    print(model)
    print("\n✅ 当前可训练参数：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")
    print(f"\nLearnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # DataLoaders
    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform,
                                image_root=args.image_root, audio_roots=args.audio_roots, audio_dims=args.audio_dims)
    if args.use_audio and train_dataset.total_audio_dim == 0:
        raise ValueError('Dataset provides no audio features but --use_audio is set')
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           collate_fn=collate_fn, num_workers=args.n_workers)

    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform,
                               image_root=args.image_root, audio_roots=args.audio_roots, audio_dims=args.audio_dims)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                          collate_fn=collate_fn, num_workers=args.n_workers)

    # Loss / Optim
    loss_fn = nn.BCELoss()  # 模型 head 已包含 Sigmoid
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_auc = 0.0

    for epoch in range(args.max_epochs):
        # -------------------- Train --------------------
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            # 自适应解包（训练通常返回 heatmaps，可能含/不含 audio）
            if len(batch) == 9:
                imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps, audio = batch
            elif len(batch) == 8:
                imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch
                audio = None
            else:
                raise ValueError(f"Unexpected train batch len={len(batch)}")

            optimizer.zero_grad()

            # 关键：将 bboxes 规整为 list[list[4]]
            safe_bboxes = _ensure_bbox_list_of_lists(bboxes)
            input_dict = {"images": imgs.cuda(), "bboxes": safe_bboxes}
            if args.use_audio and audio is None:
                raise RuntimeError('use_audio=True but audio features missing in batch')
            if args.use_audio and audio is not None:
                input_dict["audio"] = audio.cuda(non_blocking=True)

            preds = model(input_dict)
            heatmap_preds = _ensure_heatmap_BHW(preds['heatmap'])
            loss = loss_fn(heatmap_preds, heatmaps.cuda())

            if args.use_audio and hasattr(model, "audio_alpha"):
                loss = loss + args.reg_lambda * torch.norm(model.audio_alpha, p=2)

            loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                msg = f"TRAIN E{epoch} I{cur_iter}/{len(train_dl)} loss={loss.item():.4f}"
                if args.use_audio and hasattr(model, "audio_alpha"):
                    try:
                        msg += f", audio_alpha={model.audio_alpha.item():.6f}"
                    except Exception:
                        pass
                print(msg)

        scheduler.step()

        # -------------------- Eval --------------------
        model.eval()
        avg_l2s, min_l2s, aucs = [], [], []
        with torch.no_grad():
            for cur_iter, batch in enumerate(eval_dl):
                # 自适应解包（评估集通常无 heatmaps，可能含/不含 audio）
                if len(batch) == 8:
                    imgs, bboxes, gazex, gazey, inout, heights, widths, audio = batch
                elif len(batch) == 7:
                    imgs, bboxes, gazex, gazey, inout, heights, widths = batch
                    audio = None
                else:
                    raise ValueError(f"Unexpected eval batch len={len(batch)}")

                safe_bboxes = _ensure_bbox_list_of_lists(bboxes)
                input_dict = {"images": imgs.cuda(), "bboxes": safe_bboxes}
                if args.use_audio and audio is None:
                    raise RuntimeError('use_audio=True but audio features missing in eval batch')
                if args.use_audio and audio is not None:
                    input_dict["audio"] = audio.cuda(non_blocking=True)

                preds = model(input_dict)
                heatmap_preds = _ensure_heatmap_BHW(preds['heatmap'])

                B = heatmap_preds.shape[0]
                for i in range(B):
                    auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                    avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                    aucs.append(auc)
                    avg_l2s.append(avg_l2)
                    min_l2s.append(min_l2)

        if len(aucs) > 0:
            avg_auc = float(np.mean(aucs))
            print(f"EVAL E{epoch}: AUC={avg_auc:.4f}, MinL2={np.mean(min_l2s):.4f}, AvgL2={np.mean(avg_l2s):.4f}")
        else:
            avg_auc = 0.0
            print(f"EVAL E{epoch}: no valid samples")

        if args.use_audio and hasattr(model, "audio_alpha"):
            try:
                print(f"[DEBUG] audio_alpha after epoch {epoch}: {model.audio_alpha.item():.6f}")
            except Exception:
                pass

        # -------------------- Save --------------------
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }
        torch.save(ckpt_dict, os.path.join(exp_dir, f"epoch{epoch}_ckpt.pth"))
        torch.save(ckpt_dict, os.path.join(exp_dir, "last_ckpt.pth"))

        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            print(f"\n🌟 新最佳模型保存（AUC={best_auc:.4f}）")

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
