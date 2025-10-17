import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 设置为只使用 GPU 
import torch
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from data.script.fine_tuning import DualEncoder
from utils import *
import random

# 配置 int4 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # 或者 "fp4"
    bnb_4bit_compute_dtype=torch.float16  # 修改计算的数据类型为 float16
)

class ChatGPT:
    def __init__(self,role='assistant'):
        self.model_string = "gpt-3.5-turbo"
        self.api_key = ""
        self.model = ChatOpenAI(api_key=self.api_key, model=self.model_string,
                                max_tokens=4096, temperature=0.2)

        self.model = ChatOpenAI(api_key=self.api_key, model=self.model_string,
                                base_url=self.base_url, max_tokens=4096, temperature=0.2)
        
        self.role = role
        self.parser = StrOutputParser()
        self.history_message = []
        self.prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name='messages')])

    def add_history(self, message, role):
        if role == self.role:
            self.history_message.append(AIMessage(message))
        else:
            self.history_message.append(HumanMessage(message))

    def clear_history(self):
        self.history_message.clear()        
    
    def invoke(self, message):
        current_message = [HumanMessage(message)]
        formatted_message = {'messages': self.history_message + current_message}
        chain = self.prompt | self.model | self.parser
        return chain.invoke(formatted_message)

    def set_histoty(self,dialogs):
        self.clear_history()
        for utterance in dialogs:
            role,mess = utterance.split(':',1)
            self.add_history(mess,role)

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
        self.role = role

    def tokenizer_encode(self, message):
        input_ids = self.tokenizer.encode(message, max_length=8192, truncation=True, add_special_tokens=False)
        attn_mask = [1] * len(input_ids)
        return {
            'input_ids': torch.tensor(input_ids).to(self.device).unsqueeze(0),
            'attention_mask': torch.tensor(attn_mask).to(self.device).unsqueeze(0)
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
            
class Qwen:
    def __init__(self,role='assistant'):
        # 配置CUDA设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "/Data/public/Qwen2.5-7B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,                                                        
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.role = role
        self.history_message = []
        self.system_prompt = None
        
    def add_history(self, message, role):
        if role == self.role:
            self.history_message.append({"role": "assistant", "content": message})
        else:
            self.history_message.append({"role": "user", "content": message})

    def tokenizer_encode(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        return model_inputs

    def tokenizer_decode(self, generated_ids):
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    def clear_history(self):
        self.history_message.clear()   
        
    def set_histoty(self,dialogs):
        self.clear_history()
        for utterance in dialogs:
            role,mess = utterance.split(':',1)
            self.add_history(mess,role)     
    
    def invoke(self, message):
        formatted_message = []
        if self.system_prompt != None:
            formatted_message += [{"role": "system", "content": self.system_prompt}]
        current_message = [{"role": "user", "content": message}]
        formatted_message += self.history_message + current_message
        model_inputs = self.tokenizer_encode(formatted_message)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        result = self.tokenizer_decode(generated_ids)
        return result[0].strip()

model_map = {
    'Llama': Llama,
    'ChatGPT' : ChatGPT,
    'Qwen' : Qwen
} 

class Simulator_Llama(Llama):
    def __init__(self):
        super().__init__('Seeker')
        
    def set_system_prompt(self,target,seen_liked=None):
        if seen_liked != None:
            portrait = self.get_portrait(target,seen_liked)
        else:
            portrait = None
            
        self.target = re.sub(r"\s*\(.*?\)", "", target.split(': ')[0].lower())  # 去除英文括号年份及前面的空格
        
        seeker_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Please simulate acting as a "item seeker" based on the 'Seeker Information'. Your task is to conduct a dialogue recommendation simulation with the recommender to find the target item.
        In order to better simulate the real seekers' vague understanding of their target items, please summarize some vague features based on the actual target item information given to answer the recommender's related questions.
        Other Requirements:
        1. If the recommender makes a suggestion (e.g., “Is it X?”, “You might like X.”), even if it includes rich descriptions or emotional language, treat it as a recommendation only unless it contains a direct question about the item's features.
         - If it is the target item, only respond naturally to confirm.
         - If it is not the target item, only respond naturally to reject.
         - Do not volunteer your preferences or item features unless asked directly.
         - Do not ask follow-up questions.
        2. If the recommender asks about the item's features (e.g., tone, theme, genre, etc.), answer the question seriously and informatively, based on your simulated understanding of the target item.
         - Never name the target item.  
         - Avoid repeating previous info.
        3. Keep your responses short and to the point. Avoid unnecessary elaboration.
        4. If the recommender asks about information regarding the user you are simulating and you do not know the answer, you can fabricate it, but it must be consistent with the target item.
        5. Please avoid repeating information that has already been mentioned in previous responses.
        【Seeker Information:
        ·Seen-liked Sequence:{}
        ·Target Item:{}
        ·Portrait:{}】<|eot_id|>
        """
       
        self.system_prompt = seeker_prompt_template.format(seen_liked,target,portrait)
        
        
    def get_portrait(self,targets,seen_liked):
        prompt = """
        Please summarize the user's preferences based on the 'Like sequences' that the user has watched in the past and the 'Target sequences' that the user wants to watch in the future, and summarize the user's portrait in one paragraph.
        Like Sequences:{}.
        Target Sequences:{}.
        """.format(seen_liked,targets)
        return super().invoke(prompt)
        
    def invoke(self, message):
        response =  super().invoke(message)
        if self.target in message.lower():
            response += '<ACCEPTED>'
            
        return response

class Recommender:
    def __init__(self,model,num_branches):
        self.model = model_map.get(model)('Recommender')
        self.num_branches = num_branches
        self.simulator = Simulator_Llama()
        self.desc = []
        self.already_recommend_items = []
        
    def extract_need(self,feedback):
        prompt = """Extract the seeker's needs from the following seeker feedback about the movie. Please reply with a sentence describing the movie.
        【Seeker Feedback: {}】
        【Output Format: This movie is ... 】""".format(feedback)
        desc = self.model.invoke(prompt)
        return desc
    
    def get_recommended_item(self,dialogue,needs,already_recommend_items,candidate=None):        
        if candidate == None:
            prompt = """Items listed in 'Previously Recommended Items' have already been recommended before. Based on the 'Historical Dialogue' and the seeker's 'Target Item Description', recommend the single most suitable new item that fully satisfies all specified requirements, strictly excluding any item listed in 'Previously Recommended Items', and output only the name of the new recommended item.
            【Historical Dialogue: {}】
            【Target Item Description: {}】
            【Previously Recommended Items:{}】
            【Output Format: New Recommend Item: ... 】
            """.format(dialogue, needs, already_recommend_items)
        else:
            if len(candidate) > 300 :
                candidate = candidate[:300] 
            prompt = """Items listed in 'Previously Recommended Items' have already been recommended before. Based on the 'Historical Dialogue' and the seeker's 'Target Item Description', recommend the single most suitable new item from the provided 'Candidate Items' that fully satisfies all specified requirements, strictly excluding any item listed in 'Previously Recommended Items', and output only the name of the new recommended item.
            【Historical Dialogue: {}】
            【Target Item Description: {}】
            【Previously Recommended Items:{}】
            【Output Format:The New Recommend Item: ... 】
            【Candidate Items: {}】
            """.format(dialogue, needs, already_recommend_items,candidate)
        recommended_item = self.model.invoke(prompt)
        return recommended_item.split(':')[-1]
    
    def get_thoughts(self,dialogue,needs):
        prompt = """Seaker is a user who wants to find the target movie through conversation. Given the 'Historical Dialogue' and the seeker's 'Target Item Description', generate '{}' distinct and insightful thoughts focusing on different key movie features. These thoughts must make full use of the feedback signals already present in the historical dialogue, aiming to target the seeker's implicit preferences. Each thought should inspire a follow-up question that avoids broad or parallel inquiries, and instead strategically explores a previously unaddressed but relevant angle. The goal is to rapidly narrow down the candidate space and maximize the information gained for accurate recommendation.
        【Historical Dialogue: {}】
        【Target Item Description: {}】
        【Output format:
        1.Thought1: ...
        2.Thought2: ...
        3.Thought3: ...】
        """.format(self.num_branches,dialogue,needs)
        thoughts = self.model.invoke(prompt) 
        return thoughts
    
    def recommend_invoke(self, dialogue, needs, recommend_item):          
        thoughts = self.get_thoughts(dialogue, needs)
        prompt = "You are a professional recommender simulating a dialogue-based recommendation scenario with a seeker. The goal is to help the seeker identify their target item through natural interaction. I'll give you some thoughts. "
        
        if recommend_item == None:
            prompt += "For each thought, create a follow-up question or comment based on the 'Historical Dialogue'. "
        else:
            prompt += "For each given thought, combine it with the recommended item to create a separate, natural-sounding sentence. Each sentence should begin by recommending the item, then continue with the thought as a follow-up question or comment in the context of a 'historical dialogue', using a smooth transition. "
  
        prompt += """Ensure that each sentence flows naturally from the last utterance in the historical dialogue, as if it's the next turn in the conversation. Use the second-person pronoun “you” when addressing the seeker, to keep the dialogue personal and direct. Output the results formatted as a valid JSON object.
        【Historical Dialogue: {}】
        【Thoughts: {}】
        【Recommended Item:{}】
        【Output Format:{{"Response1": ..., "Response2": ..., "Response3": ...}}】
        """.format(dialogue,thoughts,recommend_item)

        res = self.model.invoke(prompt)
        return res
    
    def base_recommend_invoke(self, dialogue, candidate):
        if len(candidate) > 300:
            candidate = random.sample(candidate, 300)
        prompt = """You are a recommendation expert. Continue the conversation naturally based on the 【Historical Dialogue】. Use it to understand the seeker's preferences and speak in a consistent, conversational tone. Only recommend one item at a time from the provided 【Candidate Items】 if the seeker's preferences are clear and you’re confident it's a good match. If the preferences are unclear or insufficient, ask a thoughtful, open-ended question to learn more. Keep your reply short, natural, and focused — just like the next turn in a real conversation.
        【Historical Dialogue: {}】
        【Candidate Items: {}】
        """.format(dialogue,candidate)
        res = self.model.invoke(prompt)
        return res
    
    def init_sumulator(self,dialogue,already_recommend_items):
        prompt = """Please randomly recommend a suitable movie to the seeker based on the 'Dialogue History' of the dialogue recommendation. Please do not recommend items that have already been recommended. Please only return the name of recommended movie.
        【Dialogue History:{}】
        【Previously Recommended Items:{}】
        【Output: ...(the name of recommended movie)】
        """.format(dialogue, already_recommend_items)

        item = self.model.invoke(prompt)            
        
        augment_itemset_prompt = """
        You are a film expert. Please summarize the key features of the following movie.

        【Requirements】:
        - Write a brief and informative summary (no more than 150 words)
        - Highlight the movie’s key characteristics: genre, times, themes, emotional tone, narrative style, character focus, or setting
        - Avoid spoilers or revealing the ending
        - Use natural, professional language suitable for recommendation or cataloging

        Movie title: {}
        
        Output: {} is ...
        """.format(item,item)
        
        item_desc = self.model.invoke(augment_itemset_prompt)
        
        self.sim_targets = item + ': ' + item_desc
        self.simulator.set_system_prompt(self.sim_targets)
        
    def simulator_invoke(self,dialogue):
        self.simulator.set_histoty(dialogue[:-1])
        return self.simulator.invoke(dialogue[-1])
    
class Retrial_Model:
    def __init__(self,embedding_model_path,itemset_map):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embeding_model = DualEncoder(embedding_model_path).to(self.device)
        self.embeding_model.eval()
        # 加载需要检索的项目集以及其嵌入
        self.itemset_map = itemset_map # 嵌入需与模型保持一致
    
    def sim_evaluate(self,dialogues,needs,sim_target):
        sim_target = ['passage:' + sim_target]
        dialogues = ['query:' + ' '.join(dialogue) for dialogue in dialogues]
        needs = ['query:' + ' '.join(need) for need in needs]
        with torch.no_grad():
            utter_embedding, items_embedding = self.embeding_model(dialogues, sim_target)

        sim = torch.matmul(utter_embedding, items_embedding.T)
        
        return sim
            
    def retrial(self, dialogues, needs, all_items):
        """获得dialogues与候选集的相似度张量

        Args:
            dialogues (List(str)): 需要检索的内容

        Returns:
            Tensor : 检索内容与候选集的相似度
        """
        
        dialogues = ['query:' + ' '.join(dialogue) for dialogue in dialogues]
        needs = ['query:' + ' '.join(need) for need in needs]
        with torch.no_grad():
            if self.mode:
                utter_embedding, _ = self.embeding_model(needs, [all_items[0]])    
            else:
                utter_embedding, _ = self.embeding_model(dialogues, [all_items[0]])
        
        items_embedding = [self.itemset_map[item] for item in all_items]
        items_embedding = torch.cat(items_embedding,dim=0)
        
        logits = torch.matmul(utter_embedding.cpu(), items_embedding.T)

        output = F.softmax(logits,dim=-1)
        
        return output.argsort(descending=True).argsort() + 1
    
        
    def get_candidate(self, rank, all_items, last_rank, delta, k=0.5, ratio=0.005, a=0.3):
        """_summary_

        Args:
            rank : [N,] 当前排名
            last_rank: [N,] 上次排名
            all_items (_type_): 项目集
            delta (_type_): 排名变化
            k (_type_): 项目保留阈值

        Returns:
            _type_: _description_
        """
        N = len(rank)
        topk = int(N * ratio)
        delta = a * delta + (1 - a) * (last_rank - rank) / last_rank   # 计算排名上升比率
        topk_index = torch.where(rank <= topk)[0]
        raise_index = torch.where(delta > k)[0]
        merged_indices = torch.unique(torch.cat((topk_index, raise_index)))
        candidate = [all_items[j] for j in merged_indices]
        return candidate, delta


base_recommender_prompt = """You are a recommendation expert. Engage naturally with the seeker in a conversational tone.
Only recommend one item at a time when the seeker's preferences are clear and you're confident it's a good match.
If the information is insufficient or unclear, ask thoughtful, open-ended questions to better understand the seeker's tastes.
Do not explain or justify your recommendation, and do not show your reasoning process.
Keep your responses short, focused, and free of unnecessary elaboration.
"""
        
# baseline
class Recommender_Qwen_base(Qwen):
    def __init__(self):
        super().__init__('Recommender')
        self.system_prompt = base_recommender_prompt
        
class Recommender_chatgpt_base(ChatGPT):
    def __init__(self):
        super().__init__('Recommender')
        self.prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(base_recommender_prompt), MessagesPlaceholder(variable_name='messages')])

class Recommender_Llama_base(Llama):
    def __init__(self):
        super().__init__('Recommender')
        self.system_prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>' + base_recommender_prompt + ' <|eot_id|>'
        
    