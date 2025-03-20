import torch
import numpy as np
from sklearn import metrics
from PPlus_multilabel_bert import BERT_multilabel, validation_multilabel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义预测函数
def get_model_predictions(model_path):
    model = BERT_multilabel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    predictions, targets, _ = validation_multilabel(model)
    return np.array(predictions), np.array(targets)

# 定义模型路径列表
model_paths = [
    "../results/best_model.pt",         # 模型1: 基础模型
    "../results/model2_batch16.pt",     # 模型2: 批次大小16
    "../results/model3_arch384.pt"      # 模型3: 不同架构
]

# 收集所有模型预测
all_predictions = []
targets = None

print("加载模型预测...")
for i, path in enumerate(model_paths):
    try:
        print(f"加载模型 {i+1}: {path}")
        preds, current_targets = get_model_predictions(path)
        all_predictions.append(preds)
        if targets is None:
            targets = current_targets
    except Exception as e:
        print(f"加载模型 {i+1} 失败: {e}")

# 确认至少加载了一个模型
if not all_predictions:
    raise ValueError("没有成功加载任何模型预测")

# 集成预测 (所有模型的平均值)
ensemble_probs = np.mean(all_predictions, axis=0)

# 使用同样的阈值策略
thresholds = [0.55, 0.52, 0.40, 0.58, 0.18, 0.48, 0.52, 0.22, 0.52]
ensemble_preds = [[1 if float(prob[i]) >= thresholds[i] else 0 
                  for i in range(len(prob))] 
                 for prob in ensemble_probs]

# 计算每个单独模型的F1
print("\n单模型性能:")
for i, preds in enumerate(all_predictions):
    model_preds = [[1 if float(prob[i]) >= thresholds[i] else 0 
                  for i in range(len(prob))] 
                 for prob in preds]
    model_f1_micro = metrics.f1_score(targets, model_preds, average='micro')
    print(f"模型 {i+1} Micro F1: {model_f1_micro:.4f}")

# 评估集成模型
ensemble_f1_micro = metrics.f1_score(targets, ensemble_preds, average='micro')
ensemble_f1_macro = metrics.f1_score(targets, ensemble_preds, average='macro')

print(f"\n集成模型Micro F1: {ensemble_f1_micro:.4f}")
print(f"集成模型Macro F1: {ensemble_f1_macro:.4f}")

# 计算类别级别指标
print("\n各类别F1分数:")
class_names = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 
               'Education', 'Socioeconomic', 'Social', 'Plus']
for i, label in enumerate(class_names):
    # 提取当前类别的真实值和预测值
    true_vals = targets[:, i]
    pred_vals = [pred[i] for pred in ensemble_preds]
    f1 = metrics.f1_score(true_vals, pred_vals)
    print(f"{label}: {f1:.4f}")