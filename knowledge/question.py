import re

import requests
from jinja2 import Template
from loguru import logger

from utils.customModel import CustomModel

api_url = 'http://192.168.153.1:5050'
model_name = 'qwen/qwen3-8b'
prompt = Template(r'''
你现在需要作为专业的文本内容分析专家，基于给定的文本Chunk提取高质量的问答（QA）对，具体要求如下：

1. **数量要求**：从该Chunk中精准提取**10个独立的问答对**，不得多提或少提。
2. **问题要求**：
   - 问题需直接从Chunk的核心内容、关键信息、具体细节中衍生，避免无意义的泛化问题；
   - 问题类型需多样化，包括但不限于事实型（是什么/有哪些）、原因型（为什么）、过程型（怎么做）、比较型（有何不同）、结果型（有什么影响）等；
   - 问题需清晰、具体、可回答，且每个问题的核心指向不同，避免重复或高度相似。
3. **答案要求**：
   - 答案必须严格基于该Chunk的原文内容，**不得编造、推测或添加外部信息**；
   - 答案需简洁、准确、完整，能够直接回应问题，保留关键信息，避免冗余；
   - 若Chunk中存在数据、案例、时间、地点等具体信息，答案中需精准体现。
4. **格式要求**：
   按照以下结构化格式输出，每个问答对编号清晰：
   Q1: [具体问题]
   A1: [对应答案]
   Q2: [具体问题]
   A2: [对应答案]
   ...
   Q10: [具体问题]
   A10: [对应答案]

现在请处理以下文本Chunk：

{{chunk}}
''')


def make_request(chunk):
    payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': prompt.render(chunk=chunk)},
        ],
        "temperature": 0,
        "max_tokens": 20480,
    }
    response = requests.post(
        api_url + '/v1/chat/completions',
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    return response.json()['choices'][-1]['message']['content']


def make_question(chunk):
    llm = CustomModel(api_url=api_url, model_name=model_name, max_tokens=20480)
    question_text = llm.invoke(prompt.render(chunk=chunk))
    logger.debug(question_text)
    data = []
    for lines in question_text.split('\n\n'):
        question = re.search(r'Q\d+: (\w+)', lines).group(1)
        answer = re.search(r'A\d+: (.*)', lines).group(1)
        data.append({'question': question, 'answer': answer})
    return data


if __name__ == '__main__':
    text = make_question(r'''3.创建ROS工作空间

以建立名字为my_ws 的工作空间为例，终端输入

mkdir -p \~/my_ws/src

解压""源码""文件夹，即把 astra 相机的 Orbbec-ros-sdk 功能包文件夹复制到\~/my_ws/src下，然后输入以下命令进行编译

cd \~/my_ws   
catkin_make   
source \~/my_ws/devel/setup.bash

1）显示图像

启动相机命令:''')
    print(text)