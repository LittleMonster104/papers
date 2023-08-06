import pennylane as qml
import torch
from pennylane import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# 初始化量子设备
dev = qml.device("qiskit.aer", wires=9)  # 修改为3个量子比特

# 创建量子电路
@qml.qnode(dev)
def quantum_embedding(circuit_params, features):
    from pennylane.templates import AngleEmbedding
    from pennylane.templates.layers import StronglyEntanglingLayers
    AngleEmbedding(features, wires=range(9))
    StronglyEntanglingLayers(circuit_params, wires=range(9))

    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载Dolly的tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)

# 定义输入文本
text = "这是一个经典向量。"
print(text)

# 使用Dolly的tokenizer将文本编码为模型的输入格式
input_ids = tokenizer.encode(text, return_tensors='pt')

# 将输入文本转换为量子特征向量
features = np.array(input_ids[0][:3])
circuit_params = np.random.uniform(low=0, high=2*np.pi, size=(3, 9, 9))
print(circuit_params.shape)
circuit_params = np.transpose(circuit_params, (1, 2, 0))
# 在量子电路中计算特征向量的量子表示
quantum_result = quantum_embedding(circuit_params, features)
quantum_result = np.array(quantum_result)
# 将量子表示结果转换为经典表示
classical_result = torch.tensor(quantum_result.reshape(1, -1)).long()
# 使用Dolly模型进行情感分类
outputs = model(classical_result)
logits = outputs.logits
print(logits.shape)
# 获取情感分类结果
predicted_class = torch.argmax(torch.flatten(logits), dim=0).item()

# 打印情感分类结果
print("预测的情感分类为:", predicted_class)