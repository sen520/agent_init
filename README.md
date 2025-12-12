# agent_init

langgraph图调用简单原型

已完成：
  日志配置：日志打印定长，显示全部以及其他配置

文件：
- emb_start 手动拉起一个小模型服务
- utils
  - customModel
    - CustomModel  langgraph自定义模型服务
    - CustomEmbedding langgraph自定义embedding模型服务
  - logger

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

  