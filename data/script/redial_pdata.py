import json
import re
from functools import partial
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置为只使用 GPU 
import pandas as pd
import random
import torch.nn.functional as F
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
# from ...base_model import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import partial
import concurrent.futures
from fine_tuning import DualEncoder
path = '../redial/'

# 配置 int4 量化
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


def read_jsonl(file_path):
    # file_path = os.path.join('data/redial/', file_path)
    total_lines = sum(1 for line in open(path + file_path, 'r', encoding='utf-8'))

    with open(path + file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=total_lines, unit="line", desc="Processing lines"):
            yield json.loads(line)


def replace_keys_with_values(text, key_value_dict):
    # 编译正则表达式模式
    pattern = re.compile(r'@(\d+)')
    
    # 使用 lambda 和 functools.partial 创建回调函数
    replace_match = partial(lambda d, m: d.get(m.group(1), m.group(0)), key_value_dict)
    
    # 使用正则表达式进行替换
    text = pattern.sub(replace_match, text)
    
    return text

def preprocessing_data(file_path):
    pdata = []
    # 使用函数读取文件
    for data in read_jsonl(file_path):
        last_role = ''
        utter = ''
        utterances = []
        movie_dict = data['movieMentions']
        questions = data['initiatorQuestions']
        if not isinstance(questions,dict) or data['messages'][0]['senderWorkerId'] != data['initiatorWorkerId']:
            continue
        for message in data['messages']:
            if message['senderWorkerId'] == data['initiatorWorkerId']:
                role = 'Seeker: '
            else:
                role = 'Recommender: '
            curr_text = replace_keys_with_values(message['text'], movie_dict)
            if role == last_role:
                if utter[-1] != '?' or utter[-1] != '.':
                    utter += '.'
                utter += ' ' + curr_text
            else:
                utterances.append(utter)
                utter = role + curr_text
            last_role = role
        
        seen_liked = [] 
        targets = []
        for id,mess in data['initiatorQuestions'].items():
            if mess['liked'] == 1:
                if mess['seen'] == 1:
                    seen_liked.append(movie_dict[id])
                else:
                    targets.append(movie_dict[id])
        if len(targets) == 0:
            continue
        pdata.append({'utterances':utterances[1:],'seen_liked':seen_liked,'target':targets[0]})
    return pdata

def write_jsonl(file_path,data):
        for line in data:
            utterances = line['utterances']
            seen_liked = line['seen_liked']
            targets = line['targets']
            f.write(json.dumps({'utterances':utterances,'seen_liked':seen_liked,'targets':targets}, ensure_ascii=False) + '\n')
            

def get_itemset_embedding(itemset_info):
    file_path = path + 'itemset_map.pt'
    if os.path.exists(file_path):
        # 如果文件存在，则加载
        itemset_map = torch.load(file_path)
        
    else:
        model_name = "../checkpoint/dual_encoder_finetuned"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DualEncoder(model_name).to(device)
        itemset_map = {}
        print('Get Itemset Embedding')
        for item,info in tqdm(itemset_info.items()):
            queries = 'query:' + info
            passages = ['None']
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
    """.format(item)
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
                          
if __name__ =='__main__':
    random.seed(12)
    
    train_path = 'train_data.jsonl'
    test_path = 'test_data.jsonl'
    
    train_data = preprocessing_data(train_path)
    test_data = preprocessing_data(test_path)
    data = train_data + test_data

    itemset = set()  
    itemset_title = {}
    for one in data:
        title = one['target']
        if title not in itemset:
            itemset.add(title)
            itemset_title[title] = title
    
    df = pd.read_csv(path + 'movies_with_mentions.csv')
    itemset = set(df['movieName'].values)
    
    num_dia = len(data)
    max_turn = 0
    num_utters = 0
    num_words = 0
    num_turns = 0
    for one in data:
        utters = one['utterances']
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
        
    with open(path + 'pre_redial.jsonl', 'w', encoding='utf-8') as f:
        for one in sampled_data:
            f.write(json.dumps({'utterances':one['utterances'],'seen_liked':one['seen_liked'],'target':one['target'],'abstract':itemset_info[one['target']]}, ensure_ascii=False) + '\n')