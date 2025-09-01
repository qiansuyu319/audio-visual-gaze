#!/bin/bash
set -e

source /etc/network_turbo || echo "⚠️ 网络加速模块未找到，跳过..."

# 切换环境（优先 mamba，失败则使用 conda）
if command -v mamba >/dev/null 2>&1; then
  eval "$(mamba shell hook --shell bash)"
  mamba activate gazelle
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gazelle
else
  echo "❌ 未找到 mamba/conda，请先安装或在正确环境中运行"
  exit 1
fi


# 执行评估脚本（以模块方式运行，确保项目根目录在 PYTHONPATH 中）
python -m scripts.eval_single \
  --img_root "/root/autodl-tmp/test/frames" \
  --csv_root "/root/autodl-tmp/test/GT_CSV"
