import numpy as np
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer

def main():
    # 1. 载入模型
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 读取文本
    input_txt = 'test.txt'   # 你的文本文件
    with open(input_txt, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    # 3. 直接批量送入模型（小文本不用分批，大文本可以见上一版）
    embeddings = model.forward(texts, tokenizer)  # [N, emb_dim]
    embeddings = embeddings.cpu().numpy()

    # 4. 保存npy
    output_npy = 'test_mclip.npy'
    np.save(output_npy, embeddings)
    print(f"成功保存 {len(texts)} 条文本的embedding到 {output_npy}，shape: {embeddings.shape}")

    # 5. 简单验证输出
    for i, (text, vec) in enumerate(zip(texts, embeddings)):
        print(f"{i+1}. {text}\n  特征前5维: {vec[:5]}")
        if i >= 2:
            break

if __name__ == "__main__":
    main()
