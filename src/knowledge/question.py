import re

import requests
from loguru import logger
from jinja2 import Environment, FileSystemLoader
from src.utils.customModel import CustomModel

api_url = 'http://192.168.153.1:5050'
model_name = 'qwen/qwen3-8b'


def make_request(chunk, prompt):
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


def make_question(chunk, prompt):
    llm = CustomModel(api_url=api_url, model_name=model_name, max_tokens=20480)
    question_text = llm.invoke(prompt.render(chunk=chunk))
    logger.debug(question_text)
    data = []
    for lines in question_text.split('\n\n'):
        if re.search(r'Q\d+: (\w+)', lines) and re.search(r'A\d+: (.*)', lines):
            question = re.search(r'Q\d+: (\w+)', lines).group(1)
            answer = re.search(r'A\d+: (.*)', lines).group(1)
            data.append({'question': question, 'answer': answer})
    return data


if __name__ == '__main__':
    env = Environment(loader=FileSystemLoader('../../prompt'))
    prompt = env.get_template('qa.md')
    text = make_question(r'''3.创建ROS工作空间

以建立名字为my_ws 的工作空间为例，终端输入

mkdir -p \~/my_ws/src

解压""源码""文件夹，即把 astra 相机的 Orbbec-ros-sdk 功能包文件夹复制到\~/my_ws/src下，然后输入以下命令进行编译

cd \~/my_ws   
catkin_make   
source \~/my_ws/devel/setup.bash

1）显示图像

启动相机命令:''', prompt)
    print(text)