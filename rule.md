在这个文件中维护训练与测试的数据与权重路径。把占位路径改成你自己的绝对路径即可。

建议直接编辑下面的 YAML 配置块：

```yaml
# 路径与默认参数（示例）
train:
  img_root: /root/autodl-tmp/.autodl/train/frames
  csv_root: /root/autodl-tmp/.autodl/GT_CSV
  audio_wav_root: /root/autodl-tmp/.autodl/train/processed_features/audio_wav
  ckpt_path: /root/gazelle/scripts/experiments/vgs_egemaps_wav2vec_amp_test3/2025-08-21_17-27-10/epoch_9.pt

test:
  img_root: /root/autodl-tmp/test/frames
  csv_root: /root/autodl-tmp/test/GT_CSV
  audio_wav_root: /root/autodl-tmp/test/audio
  ckpt_path: /root/gazelle/scripts/experiments/vgs_egemaps_wav2vec_amp_test3/2025-08-21_17-27-10/epoch_9.pt

defaults:
  model_name: gazelle_dinov2_vitb14
  batch_size: 64
  audio_fps: 25
  audio_win_sec: 5