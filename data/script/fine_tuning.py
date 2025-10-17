import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import os
import json
from tqdm import tqdm

torch.manual_seed(42)

# 构造自定义数据集，数据格式为 (dialog_text, candidate_text) 对
class RetrievalDataset(Dataset):
    def __init__(self, data):
        """
        data: 列表，每个元素为 (dialog, passage) 对，
              dialog 例如："query: Seeker: ...", passage 例如："passage: V for Vendetta (2005) is ..."
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# 定义双塔模型，分别对对话（query）和候选 passage 进行编码
class DualEncoder(nn.Module):
    def __init__(self, model_name):
        super(DualEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, queries, passages):
        # 编码对话文本
        query_inputs = self.tokenizer(queries, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=512)
        passage_inputs = self.tokenizer(passages, 
                                        return_tensors="pt", 
                                        padding=True, 
                                        truncation=True, 
                                        max_length=512)
        # 将输入数据发送到模型所在设备
        query_inputs = {k: v.to(self.encoder.device) for k, v in query_inputs.items()}
        passage_inputs = {k: v.to(self.encoder.device) for k, v in passage_inputs.items()}

        # 获取最后一层隐藏状态输出
        q_outputs = self.encoder(**query_inputs)
        p_outputs = self.encoder(**passage_inputs)
        # 采用均值池化生成固定维度向量
        q_emb = q_outputs.last_hidden_state.mean(dim=1)
        p_emb = p_outputs.last_hidden_state.mean(dim=1)
        return q_emb, p_emb

# 定义对比损失函数，采用 in-batch negatives 策略
def compute_contrastive_loss(q_emb, p_emb, temperature=0.05):
    # 计算相似度矩阵，归一化温度用于平滑分布
    sim_matrix = torch.matmul(q_emb, p_emb.T) / temperature
    # 每一行的正样本为对角线位置
    labels = torch.arange(q_emb.size(0)).to(q_emb.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def processing_data(data):
    pre_data = []

    for one in data:
        dialog = 'query:' + ' '.join(one['dialog'])
        itemset = 'passage:' + one['target']
        pre_data.append((dialog,itemset))
            
    return pre_data
        
def fine_tune():
    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "intfloat/e5-large-v2"
    model = DualEncoder(model_name).to(device)

    # 数据
    train_path = '../fine_tune_data/Llama_fine_tune_data.jsonl'
    with open(train_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
    
    train_data = processing_data(data[:-500])
    valid_data = processing_data(data[-500:])
   
    train_dataset = RetrievalDataset(train_data)
    valid_dataset = RetrievalDataset(valid_data)
    
    
    # 训练设置
    batch_size = 16 # 根据GPU显存可适当减小
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: list(zip(*batch))) # collate_fn 这里简单地返回分别为 queries 与 passages 的列表
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: list(zip(*batch)))
    
    epochs = 20
    lr = 2e-5
    patience = 3
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(0.1 * total_steps), 
                                                num_training_steps=total_steps)

    # 开始训练
    best_acc = 0
    counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for queries, passages in tqdm(train_dataloader):
            optimizer.zero_grad()
            q_emb, p_emb = model(queries, passages)
            loss = compute_contrastive_loss(q_emb, p_emb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        model.eval()
        acc = 0
        with torch.no_grad():
            for queries, passages in valid_dataloader:
                q_emb, p_emb = model(queries, passages)
                sim_matrix = torch.matmul(q_emb, p_emb.T)
                pre = torch.argmax(sim_matrix,dim=-1)
                label = torch.arange(q_emb.size(0)).to(sim_matrix.device)
                acc += (pre == label).sum().item() 
        acc /= 100
        print(f'Epoch {epoch+1} Valid Acc:{acc}')

        # 将微调后的模型保存到本地,只保留历史最优的模型
        if acc > best_acc:
            counter = 0
            best_acc = acc
            save_path = f"../checkpoint/dual_encoder_finetuned"
            model.encoder.save_pretrained(save_path)
            model.tokenizer.save_pretrained(save_path)
            print(f"微调后的模型已保存到 {save_path}")
        else:
            counter += 1
            if counter >= patience:
                print("acc连续三个epoch没有提升，提前结束")
                break
            
    return model

if __name__ == '__main__':
    fine_tune()
    
