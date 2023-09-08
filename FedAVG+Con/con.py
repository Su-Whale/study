import torch
import torch.nn.functional as F

# 定义两个向量
vector1 = torch.tensor([1.0, 2.0, 3.0])
vector2 = torch.tensor([4.0, 5.0, 6.0])

# 计算余弦相似度
cosine_similarity = F.cosine_similarity(vector1, vector2, dim=0)

print("余弦相似度:", cosine_similarity.item())
