import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import re



def only_letters_and_spaces(text):
    # 使用正则表达式，只保留字母、空格、数字
    cleaned_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return cleaned_text


def format_str(s):
    """
    确保字符串以'['开头和']'结尾。如果不是，则添加相应的方括号。

    参数:
        s (str): 需要检查并可能添加方括号的字符串。

    返回:
        str: 修改后的字符串，确保以'['开头和']'结尾。
    """
    s = s.replace('\n','')
    bracket_pos = s.rfind('{')  # 查找最后一个'['的位置
    if bracket_pos == -1:
        start = '{' # 如果没有找到'['，返回原始字符串或根据需要返回空字符串
    else:
        # 使用切片操作截取从'['开始到结尾的部分
        start = ''
        s =  s[bracket_pos:]
    bracket_pos = s.rfind('}')
    if bracket_pos == -1:
        end = '}' # 如果没有找到'['，返回原始字符串或根据需要返回空字符串
    else:
        # 使用切片操作截取从'['开始到结尾的部分
        end = ''
        s =  s[:bracket_pos+1]
    return f"{start}{s}{end}"

