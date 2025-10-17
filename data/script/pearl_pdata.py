import json
import torch.nn.functional as F
import os
import random
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from functools import partial
import concurrent.futures
from fine_tuning import DualEncoder
path = '../pearl/'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # 或者 "fp4"
    bnb_4bit_compute_dtype=torch.float16  # 修改计算的数据类型为 float16
)


class Llama:
    def __init__(self, role='assistant'):
        # 配置CUDA设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型并直接移动到CUDA设备
        self.model = AutoModelForCausalLM.from_pretrained('/Data/public/Llama-3.1-8B-Instruct', 
                                                        quantization_config=bnb_config,
                                                        torch_dtype=torch.float16).to(self.device)
        # self.model = AutoModelForCausalLM.from_pretrained('../meta-llama/Llama-3.1-8B-Instruct', device_map='auto',
        #                                                   quantization_config=bnb_config,torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained('/Data/public/Llama-3.1-8B-Instruct')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.history_message = ''
        self.system_prompt = ''
        # self.model.eval()
        self.role = role

    def tokenizer_encode(self, message):
        input_ids = self.tokenizer.encode(message, max_length=4096, truncation=True, add_special_tokens=False)
        attn_mask = [1] * len(input_ids)
        return {
            'input_ids': torch.tensor(input_ids).unsqueeze(0).to(self.device),
            'attention_mask': torch.tensor(attn_mask).unsqueeze(0).to(self.device)
        }

    def tokenizer_decode(self, tokens, input_length):
        return self.tokenizer.batch_decode(tokens[:, input_length:], skip_special_tokens=True)

    def invoke(self, message):
        formatted_message = self.system_prompt
        formatted_message += self.history_message
        formatted_message += '<|start_header_id|>user<|end_header_id|>\n\n'
        formatted_message += message
        formatted_message += '<|eot_id|>\n'
        formatted_message += '<|start_header_id|>assistant<|end_header_id|>\n\n'
        tokens = self.tokenizer_encode(formatted_message)
        input_length = tokens['input_ids'].shape[1]
        
        with torch.no_grad():
            output = self.model.generate(**tokens, max_new_tokens=1024, do_sample=True,
                                         pad_token_id=self.tokenizer.eos_token_id)  # , temperature=0.2)
        result = self.tokenizer_decode(output, input_length)
        return result[0].strip()

    def add_history(self, message, role):
        if role == self.role:
            self.history_message += '<|start_header_id|>assistant<|end_header_id|>\n\n'
            self.history_message += message
            self.history_message += '<|eot_id|>\n'
        else:
            self.history_message += '<|start_header_id|>user<|end_header_id|>\n\n'
            self.history_message += message
            self.history_message += '<|eot_id|>\n'
            
    def clear_history(self):
        self.history_message = ''

    def set_histoty(self,dialogs):
        self.clear_history()
        for utterance in dialogs:
            role,mess = utterance.split(':',1)
            self.add_history(mess,role)

def average_pool(last_hidden_states: Tensor,attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
             
def get_itemset_embedding(itemset_info):
    file_path = path + 'itemset_map.pt'
    if os.path.exists(file_path):
        # 如果文件存在，则加载
        itemset_map = torch.load(file_path)
        
    else:
        model_name = "../checkpoint/dual_encoder_finetuned"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.device_count()) 
        model = DualEncoder(model_name).to(device)
        itemset_map = {}
        print('Get Itemset Embedding')
        for item,info in tqdm(itemset_info.items()):
            queries = 'query:' + info
            passages = ['None'] # 不需要其输出，设为None
            with torch.no_grad():
                # Tokenize the input texts
                embeddings, _ = model(queries, passages)
            itemset_map[item] = embeddings.cpu()
        torch.save(itemset_map, file_path)
    return itemset_map

def preccesed(item,model):
    augment_itemset_prompt = """
    You are a film expert. Please summarize the key features of the following movie.

    【Requirements】:
    - Write a brief and informative summary (no more than 150 words)
    - Highlight the movie’s key characteristics: genre, times, themes, emotional tone, narrative style, character focus, or setting
    - Avoid spoilers or revealing the ending
    - Use natural, professional language suitable for recommendation or cataloging

    Movie title: {}
    
    Your Response: {} is ...
    """.format(item,item)
    info = item + model.invoke(augment_itemset_prompt)
    return item, info

def augment_itemset(itemset):
    file_path = path + 'itemset_info.pt'
    if os.path.exists(file_path):
        # 如果文件存在，则加载
        itemset_info = torch.load(file_path)
    else:   
        itemset = list(itemset)
        model = Llama()
        partial_preccesed = partial(preccesed, model=model)
        itemset_info = {}
        size = len(itemset)
        step = int(size / 10) + 1 
        for i in range(10):
            # 创建线程池
            start = i*step
            end = (i+1)*step if (i+1)*step < size else size
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # 提交任务到线程池
                itemset_info.update(dict(tqdm(executor.map(partial_preccesed, itemset[start:end]), total=(end-start))))
            torch.save(itemset_info, file_path)
    return itemset_info

if __name__ == '__main__':
    random.seed(12)
    data = []
    with open(path + "train.json", "r", encoding="utf-8") as f: 
        data.extend(json.load(f))
    with open(path + "valid.json", "r", encoding="utf-8") as f: 
        data.extend(json.load(f))
    with open(path + "test.json", "r", encoding="utf-8") as f: 
        data.extend(json.load(f))
        
    itemset = set()  
    # itemset_info = {}
    itemset_title = {}
    for one in data:
        title = one['gt_movie_title']
        if title not in itemset:
            itemset.add(title)
            itemset_title[title] = title
    
    num_dia = len(data)
    max_turn = 0
    num_utters = 0
    num_words = 0
    num_turns = 0
    for one in data:
        utters = one['dialogue']
        num_utters += len(utters)
        turn = len(utters) // 2
        num_turns += turn
        max_turn = turn if turn > max_turn else max_turn
        for utter in utters:
            num_words += len(utter.split(' '))
    print(f"#Dialogues: {num_dia}  #Items:{len(itemset)}  #Utterances: {num_utters}  Avg.turns: {num_turns/num_dia}  Max.turns: {max_turn}  Avg.words/utterance: {(num_words/num_utters)-1}")
    
    itemset_info = augment_itemset(itemset)
    itemset_map = get_itemset_embedding(itemset_info)
    
    if len(data) >= 100: 
        sampled_data = random.sample(data, 100)
    else: 
        sampled_data = data
        
    cur = {}
    with open(path + 'pre_pearl.jsonl', 'w', encoding='utf-8') as f:
        for one in sampled_data:
            if one['gt_movie_title'] == "":
                continue
            while one['dialogue'] and not one['dialogue'][0].startswith('Seeker'): 
                one['dialogue'].pop(0)
            cur['utterances'] = one['dialogue']
            cur['seen_liked'] = one['seen_movie_titles']
            cur['target'] =  one['gt_movie_title']
            cur['abstract'] = itemset_info[cur['target']]
            f.write(json.dumps(cur, ensure_ascii=False) + '\n')
    