from base_models import *
import os
import json
from tqdm import tqdm
import argparse
from datetime import datetime

model_map = {
    'Llama': Recommender_Llama_base,
    'ChatGPT' : Recommender_chatgpt_base,
    'Qwen' : Recommender_Qwen_base
}
    

def run_base(recommender_model,seeker_model,line,turn):
    # 初始化模型
    target = line['target']
    seen_liked = line['seen_liked']
    seeker_model.set_system_prompt(target,seen_liked)
    
    dialog = [line['utterances'][0]]

    is_successfull = 0
    with tqdm(total=turn) as pbar:
        for t in range(turn):
            pbar.set_postfix_str(f'Turn {t+1}')
            pbar.update(1)
            recommender_model.set_histoty(dialog[:-1])
            # 调用推荐模型并获取响应
            recommender_responses = recommender_model.invoke(dialog[-1].split(':',1)[1])
            dialog.append('Recommender: ' + recommender_responses)
            seeker_model.set_histoty(dialog[:-1]) #最后一句话作为输入，而不是历史
            seeker_response = seeker_model.invoke(dialog[-1].split(':',1)[1])
            dialog.append('Seeker: ' + seeker_response) 
            if '<ACCEPTED>' in seeker_response: # 如果该路径已成功，
                # successful_dialog.append(dialog)
                # seeker_response = seeker_model.check_accepted(dialog[-1],targets)
                is_successfull = 1
                break 
    
    return dialog, is_successfull, t+1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='redial')
    parser.add_argument("--baseline", type=str, default='ChatGPT')

    args = parser.parse_args()
    
    dataset = args.dataset
    baseline = args.baseline
    turn = 5
    
    with open(os.path.join(f'data/{dataset}/pre_{dataset}.jsonl')) as f:
            datas = f.readlines()

    seeker_model = Simulator_Llama()
    recommender_model = model_map.get(baseline)()
    
    seeker_model.clear_history()
    recommender_model.clear_history()
    
    
    # 获取当前日期和时间
    current_datetime = datetime.now()
    success_rate = 0
    average_turn = 0
    with open(os.path.join(f'baseline/output/{dataset}/',f'{baseline}_base_result_{current_datetime}'), 'w', encoding='utf-8') as f:
        for i,line in enumerate(datas):
            line = json.loads(line)
            print(f"Sample {i+1} Solution:")
            dialog,is_successfull,used_turn = run_base(recommender_model,seeker_model,line,turn)
            success_rate += is_successfull
            average_turn += used_turn
            print({'dialog':dialog,'is_successfull':is_successfull,'target':line['target']})
            f.write(json.dumps({'dialog':dialog,'is_successfull':is_successfull,'target':line['target']}, ensure_ascii=False) + '\n')
        f.write(json.dumps({'average_turn':average_turn/(i+1),'success_rate':success_rate/(i+1)}, ensure_ascii=False) + '\n')