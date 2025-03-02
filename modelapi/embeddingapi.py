from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Union
import numpy as np

app = Flask(__name__)

# 加载模型和tokenizer
# MODEL_PATH = "/Users/penn/models/bge-large-zh-v1.5"
MODEL_PATH = "/Users/johnllm/llms/bge-large-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_embeddings(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    获取文本的embedding向量
    """
    # 确保输入是列表格式
    if isinstance(texts, str):
        texts = [texts]
    
    # 添加特殊前缀
    texts = [f"为这个句子生成表示：{text}" for text in texts]
    
    # 对文本进行编码
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # 将输入移到对应设备
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # 获取embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # 转换为numpy数组并返回
    return embeddings.cpu().numpy().tolist()

@app.route('/embed', methods=['POST'])
def embed():
    """
    接收POST请求，返回文本的embedding向量
    """
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': '请提供texts字段'}), 400
        
        texts = data['texts']
        embeddings = get_embeddings(texts)
        
        return jsonify({
            'success': True,
            'embeddings': embeddings
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9550) 