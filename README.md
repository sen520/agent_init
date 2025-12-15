# agent_init

langgraph图调用简单原型

已完成：
  日志配置：日志打印定长，显示全部以及其他配置
  知识库解析成md, chunk切分

文件：
- emb_start 手动拉起一个小模型服务
- utils
  - customModel
    - CustomModel  langgraph自定义模型服务
    - CustomEmbedding langgraph自定义embedding模型服务
  - logger
- knowledge
  - converted 放置转化结果
  - files 待转化的文件
  - tmp office转pdf的结果
  - chunk chunk分割
    - 基于Token Size的分割  
    - 基于Markdown递归分割 
    - 基于语义的分割结果
  - convert mineru转化
- convert 各种文档转markdown, 暂支持 pdf, office, png等图片, zip

todo
- prompt抽出单个文件
- 模块分离
  - config/ 配置
  - state/ state结构，schema
  - tools/ 工具定义，工具初始化
  - nodes/ 节点定义
  - graph/ 图定义，组装节点
  - script/ 执行
  - logs/ 日志
  - knowledge/ 知识库处理

  

## mineru安装
参考 https://github.com/opendatalab/MinerU


pip install uv -i https://mirrors.aliyun.com/pypi/simple
uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple 

配置torch

- pip uninstall pytorch
- pip uninstall torchvision
- pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

模型下载位置 C:\Users\admin\.cache\modelscope

mineru.exe -p '.\0. 树莓派5新手入门手册.pdf' -o . --source modelscope

支持文件
```python
pdf_suffixes = ['.pdf']
office_suffixes = ['.ppt', '.pptx', '.doc', '.docx', 'xls', 'xlsx'] # 需要安装libreoffice【win, linux】, doc.spire【win】
image_suffixes = ['.png', '.jpeg', '.jpg']
compress = ['.zip', '.rar']

```