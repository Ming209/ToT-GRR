from base_models import *
import os
import json
from tqdm import tqdm
from datetime import datetime
import torch
from utils import *
import yaml
import argparse

num_map = {1:'one',2:'two',3:'three',4:'four',5:'five'}

# 广度优先模拟
def bfs_simulate(recommender_model,retrial_model,combined_data,NUM_SIM,MAX_RETRIES):
    root_dialogue = combined_data[0]
    already_recommend_items = combined_data[3][:]
    recommender_model.init_sumulator(root_dialogue,already_recommend_items)
    succes_sim = True
    sim_turn = 0
    combined_datas = [combined_data]
    best_recommender_response = None
    while sim_turn < NUM_SIM:
        success_batches = []
        for dialogue,needs,recommend_item,already_recommend_items in combined_datas:
            retries1 = 0
            while retries1 < MAX_RETRIES:
                try:
                    # 调用推荐模型并获取响应
                    r_responses = recommender_model.recommend_invoke(dialogue,needs,recommend_item)
                    # 尝试解析JSON响应
                    recommender_responses = json.loads(format_str(r_responses))
                    if len(recommender_responses) != recommender_model.num_branches:
                        raise ValueError(f"Expected {recommender_model.num_branches} responses, but got {len(recommender_responses)}.")
                    already_recommend_items.append(recommend_item)
                    one_batch = []
                    for _,recommender_response in recommender_responses.items():
                        seeker_response = recommender_model.simulator_invoke(dialogue + [recommender_response])
                        d = dialogue + ['Recommender: ' + recommender_response] + ['Seeker: ' + seeker_response]
                        need = recommender_model.extract_need(d[-1])
                        item = recommender_model.get_recommended_item(d,needs+[need],already_recommend_items)
                        combined_data = (d, needs+[need],item, already_recommend_items[:])
                        one_batch.append(combined_data)

                        if '<ACCEPTED>' in seeker_response and  sim_turn != 0:
                            # 计算分支下的accepted数量
                            best_recommender_response = d[-2-2*sim_turn]
                            # accepted_count[best_recommender_response] += 1
                            
                    success_batches.extend(one_batch)  
                    # for key,value in recommender_responses:
                    #     dialogues_tree.add_edge(dialogue[-1], 'recommender: ' + value)
                    break  # 如果成功解析，则退出循环                
                except (json.JSONDecodeError,ValueError,TypeError,IndexError) as e:
                    print(f"An error occurred: {e}. 尝试重新运行 ({retries1 + 1}/{MAX_RETRIES})...")
                    retries1 += 1
                    if retries1 == MAX_RETRIES:
                        print("达到最大重试次数，放弃该对话")
            
            if best_recommender_response is not None:
                break
            
        if len(success_batches) != 0:
            combined_datas = success_batches
            sim_turn += 1
        else:
            succes_sim = False
            break # 全部失败，此次模拟作废
    
    if succes_sim:
        if best_recommender_response is not None:
            root_dialogue.append(best_recommender_response)
        else:
            dialogues = [data[0] for data in combined_datas]
            needs = [data[1] for data in combined_datas]
            output= retrial_model.sim_evaluate(dialogues,needs,recommender_model.sim_targets)
            index = torch.argmax(output)
            root_dialogue = dialogues[index][:-2*NUM_SIM+1] 
    
    return root_dialogue,succes_sim

def simulate_once(combined_data,MAX_RETRIES):
    succes_sim = False
    best_recommender_response = None
    combined_datas = []
    
    dialogue,needs,recommend_item,already_recommend_items = combined_data

    retries1 = 0
    while retries1 < MAX_RETRIES:
        try:
            # 调用推荐模型并获取响应
            r_responses = recommender_model.recommend_invoke(dialogue,needs,recommend_item)
            # 尝试解析JSON响应
            recommender_responses = json.loads(format_str(r_responses))
            if len(recommender_responses) != recommender_model.num_branches:
                raise ValueError(f"Expected {recommender_model.num_branches} responses, but got {len(recommender_responses)}.")
            already_recommend_items.append(recommend_item)
            one_batch = []
            for _,recommender_response in recommender_responses.items():
                seeker_response = recommender_model.simulator_invoke(dialogue + [recommender_response])
                d = dialogue + ['Recommender: ' + recommender_response] + ['Seeker: ' + seeker_response]
                need = recommender_model.extract_need(d[-1])
                item = recommender_model.get_recommended_item(d,needs+[need],already_recommend_items)
                combined_data = (d, needs+[need],item, already_recommend_items[:])
                one_batch.append(combined_data)

                if '<ACCEPTED>' in seeker_response:
                    best_recommender_response = d
                    
            break  # 如果成功解析，则退出循环                
        except (json.JSONDecodeError,ValueError,TypeError,IndexError) as e:
            print(f"An error occurred: {e}. 尝试重新运行 ({retries1 + 1}/{MAX_RETRIES})...")
            retries1 += 1
            if retries1 == MAX_RETRIES:
                one_batch = []
                print("达到最大重试次数，放弃该对话")

    if len(one_batch) != 0:
        combined_datas = one_batch
        succes_sim = True
    
    return combined_datas,best_recommender_response,succes_sim
    
# # 深度优先模拟                
def dfs_simulate(recommender_model,retrial_model,combined_data,MAX_SIM,MAX_RETRIES):
    root_dialogue = combined_data[0]
    already_recommend_items = combined_data[3][:]
    recommender_model.init_sumulator(root_dialogue,already_recommend_items)
    succes_sim = True
    sim_turn = 1
    combined_datas = [combined_data]
    best_recommender_response = None

    combined_datas,best_recommender_response,succes_sim = simulate_once(combined_data,MAX_RETRIES)
    cand_datas = combined_datas
    
    if best_recommender_response is not None:
        root_dialogue.append(best_recommender_response[-2])
        return root_dialogue,succes_sim
    
    while sim_turn < MAX_SIM:
        next_cand = []
        for combined_data in cand_datas:            
            combined_datas,best_recommender_response,succes_sim = simulate_once(combined_data,MAX_RETRIES)    
            
            if not succes_sim:
                continue
            
            if best_recommender_response is not None:
                root_dialogue.append(best_recommender_response[-2-2*sim_turn])
                return root_dialogue,succes_sim
            else:
                dialogues = [data[0] for data in combined_datas]
                needs = [data[1] for data in combined_datas]
                output= retrial_model.sim_evaluate(dialogues,needs,recommender_model.sim_targets)
                index = torch.argmax(output)
                next_cand.append(combined_datas[index])
                # root_dialogue = dialogues[index][:-2*NUM_SIM+1] 
            
        cand_datas = next_cand
        sim_turn += 1
    
    if len(cand_datas) != 0:
        succes_sim = True
        # 所有候选都相同，默认选第一个
        root_dialogue = cand_datas[0][0][:-2*MAX_SIM+1] 
        
    return root_dialogue,succes_sim
    
   
def tot_grr(recommender_model,seeker_model,line, retrial_model, config):
    kappa = config['kappa']
    lambda_ = config['lambda_']
    gamma = config['gamma']
    MAX_RETRIES = config['MAX_RETRIES']
    NUM_SIM = config['NUM_SIM']
    MAX_SIM = config['MAX_SIM']
    
    turn = config['turn']
    search = config['search']      
    
    # 初始化模型
    target = line['target'] 
    target_desc = target + ": " + line['abstract']
    seen_liked = line['seen_liked']
    seeker_model.set_system_prompt(target_desc,seen_liked)
    
    # 初始化候选集
    itemset = set(retrial_model.itemset_map.keys())
    all_items = ([target] + list(itemset - set(target)))
    candidate = all_items
    
    # 初始化对话数据
    needs = []
    recommended_item = None
    dialogue = [line['utterances'][0]]
    already_recommend_items = []
    combined_data = (dialogue,needs,recommended_item,already_recommend_items[:])
    

    
    last_rank =  retrial_model.retrial([dialogue], [needs], all_items)[0]
    delta = torch.zeros_like(last_rank) + kappa
    
    t = 0
    is_successfull = 0
    with tqdm(total=turn) as pbar:
        while t < turn :
            if is_successfull:
                break
            pbar.set_postfix_str(f'Turn {t+1}')
            retries2 = 0
            if search == 'bfs':
                dialogue,succes_sim = bfs_simulate(recommender_model,retrial_model,combined_data,NUM_SIM,MAX_RETRIES)
            else:
                dialogue,succes_sim = dfs_simulate(recommender_model,retrial_model,combined_data,MAX_SIM,MAX_RETRIES)
            already_recommend_items.append(combined_data[2])
                    
            if retries2 < MAX_RETRIES and succes_sim: 
                # seeker进行回复
                last_utter = dialogue[-1]
                seeker_model.set_histoty(dialogue[:-1]) #最后一句话作为输入，而不是历史
                seeker_response = seeker_model.invoke(last_utter.split(':',1)[1])
                dialogue = dialogue + ['Seeker: ' + seeker_response]
                need = recommender_model.extract_need(dialogue[-1])
                needs.append(need)
                
                rank = retrial_model.retrial([dialogue],[needs], all_items)[0]
                candidate, delta = retrial_model.get_candidate(rank, all_items, last_rank, delta, kappa, gamma, lambda_)
                
                # use candidate
                recommended_item = recommender_model.get_recommended_item(dialogue,needs,already_recommend_items,candidate)
                
                # not use candidate
                # recommended_item = recommender_model.get_recommended_item(dialogue,needs,already_recommend_items)
                
                combined_data = (dialogue,needs[:],recommended_item,already_recommend_items[:])
                t += 1
                pbar.update(1)
                
                if '<ACCEPTED>' in seeker_response: # 如果该路径已成功，
                    is_successfull = 1
                    break      
                
                print(len(candidate),candidate[0])
                                            
        
    return dialogue, needs, is_successfull, t # 取第一个对话为失败对话


if __name__ == "__main__":
    # 1. 读取配置文件
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 2. 定义命令行参数解析
    parser = argparse.ArgumentParser(description="Run experiment with configurable dataset, base model, and search method.")

    parser.add_argument("--dataset", type=str, default=config.get('dataset'),
                        help="Dataset name, overrides config.yaml if specified.")
    parser.add_argument("--base_model", type=str, default=config.get('base_model'),
                        help="Base model name, overrides config.yaml if specified.")
    parser.add_argument("--search", type=str, default=config.get('search'),
                        help="Search method, overrides config.yaml if specified.")

    # 3. 解析命令行参数
    args = parser.parse_args()

    # 4. 将参数整合
    dataset = args.dataset
    base_model = args.base_model
    search = args.search 
    
    # 5. 其他配置仍来自 config.yaml
    num_tot_candidates = config['num_tot_candidates']
    embedding_model_path = config['embedding_model_path']
    itemset_embedding_path = config['itemset_embedding_path']
    
    with open(os.path.join(f'data/{dataset}', f'pre_{dataset}.jsonl')) as f:
            datas = f.readlines()    
    
    seeker_model = Simulator_Llama()
    recommender_model = Recommender(base_model,num_tot_candidates)
    seeker_model.clear_history()
    recommender_model.model.clear_history()
    
    itemset_map = torch.load(itemset_embedding_path)

    retrial_model = Retrial_Model(embedding_model_path,itemset_map)
    
    # 获取当前日期和时间
    current_datetime = datetime.now()
    success_rate = 0
    average_turn = 0
    success_rate_5 = 0
    
    with open(os.path.join(f'output/{dataset}/',f'ToT_GRR_{base_model}_{search}_{current_datetime}'), 'w', encoding='utf-8') as f:
        for i,line in enumerate(datas):
            line = json.loads(line)
            print(f"Sample {i+1} Solution:")            
            dialogue, needs, is_successfull, used_turn = tot_grr(recommender_model,seeker_model,line, retrial_model)
            if used_turn <= 5:
                success_rate_5 += is_successfull
            average_turn += used_turn
            print({'dialogue':dialogue,'is_successfull':is_successfull,'targets':line['target']})
            f.write(json.dumps({'dialogue':dialogue,'needs':needs,'is_successfull':is_successfull,'used_turn':used_turn,'target':line['target']}, ensure_ascii=False) + '\n')
        f.write(json.dumps({'average_turn':average_turn/(i+1),'success_rate_5':success_rate_5/(i+1)}, ensure_ascii=False) + '\n')


