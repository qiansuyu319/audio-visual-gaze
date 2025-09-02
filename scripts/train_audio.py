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

# Enable cuDNN benchmark mode for faster training with fixed input shapes
torch.backends.cudnn.benchmark = True


# ------------------------- Utilities -------------------------
def _ensure_heatmap_BHW(h):
    """Ensure heatmap has shape [B, H, W]."""
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
    """Ensure bounding boxes are always in List[List[box]] format."""
    out = []
    for b in bboxes:
        if isinstance(b, (list, tuple)) and len(b) > 0 and isinstance(b[0], (list, tuple, torch.Tensor)):
            out.append(b)
        else:
            out.append([b])
    return out


# ------------------------- Training -------------------------
def main(args):
    # Experiment save directory
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir, exist_ok=True)

    # Compute total audio feature dimension
    total_audio_dim = sum(args.audio_dims) if args.use_audio and args.audio_dims else 0
    if args.use_audio and total_audio_dim == 0:
        raise ValueError('--use_audio requires valid --audio_dims')

    # Build model
    model, transform = get_gazelle_model(args.model, use_audio=args.use_audio, audio_dim=total_audio_dim)
    model.cuda()

    # Load visual-only checkpoint
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        print(f"\n🔄 Loading checkpoint from: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)  # strict=False allows missing audio branch
    else:
        print("\nℹ️ No --resume_ckpt provided; training from scratch.")

    # Freeze visual backbone (only train audio branch and new heads)
    for param in model.backbone.parameters():
        param.requires_grad = False

    print("\n✅ Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f" - {name}")
    print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Build datasets
    train_dataset = GazeDataset(
        'gazefollow', args.data_path, 'train', transform,
        image_root=args.image_root, audio_roots=args.audio_roots, audio_dims=args.audio_dims
    )
    eval_dataset = GazeDataset(
        'gazefollow', args.data_path, 'test', transform,
        image_root=args.image_root, audio_roots=args.audio_roots, audio_dims=args.audio_dims
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.n_workers
    )
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.n_workers
    )

    # Loss, optimizer, and scheduler
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_auc = 0.0

    # ------------------------- Training Loop -------------------------
    for epoch in range(args.max_epochs):
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            if len(batch) == 9:
                imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps, audio = batch
            elif len(batch) == 8:
                imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch
                audio = None
            else:
                raise ValueError(f"Unexpected train batch len={len(batch)}")

            optimizer.zero_grad()

            safe_bboxes = _ensure_bbox_list_of_lists(bboxes)
            input_dict = {"images": imgs.cuda(), "bboxes": safe_bboxes}
            if args.use_audio and audio is not None:
                input_dict["audio"] = audio.cuda(non_blocking=True)

            preds = model(input_dict)
            heatmap_preds = _ensure_heatmap_BHW(preds['heatmap'])
            loss = loss_fn(heatmap_preds, heatmaps.cuda())

            # Regularization on audio branch (if applicable)
            if args.use_audio and hasattr(model, "audio_alpha"):
                loss += args.reg_lambda * torch.norm(model.audio_alpha, p=2)

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

        # ------------------------- Evaluation -------------------------
        model.eval()
        avg_l2s, min_l2s, aucs = [], [], []
        with torch.no_grad():
            for batch in eval_dl:
                if len(batch) == 8:
                    imgs, bboxes, gazex, gazey, inout, heights, widths, audio = batch
                elif len(batch) == 7:
                    imgs, bboxes, gazex, gazey, inout, heights, widths = batch
                    audio = None
                else:
                    raise ValueError(f"Unexpected eval batch len={len(batch)}")

                safe_bboxes = _ensure_bbox_list_of_lists(bboxes)
                input_dict = {"images": imgs.cuda(), "bboxes": safe_bboxes}
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

        # ------------------------- Save Checkpoints -------------------------
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
            print(f"\n🌟 New best model saved (AUC={best_auc:.4f})")


# ------------------------- Entry Point -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14")
    parser.add_argument('--data_path', type=str, default='/root/audio-visual-gaze/train/frames')
    parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
    parser.add_argument('--exp_name', type=str, default='train_gazefollow_audio')
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--image_root', type=str, default=None)

    # Audio
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--audio_roots', type=str, nargs='+', default=None)
    parser.add_argument('--audio_dims', type=int, nargs='+', default=None)
    parser.add_argument('--reg_lambda', type=float, default=1e-5)

    # Resume from pretrained visual checkpoint
    parser.add_argument('--resume_ckpt', type=str, default='/root/gazelle_dinov2_vitb14.pt')

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    main(args)
