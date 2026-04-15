# Volcengine RAG Node for ComfyUI

基于字节跳动知识库 API 的 RAG 检索和 LLM 生成节点。

## 安装

1. 把 `comfyui_volcengine_rag_node.py` 放到 ComfyUI 的 `custom_nodes` 文件夹

2. 安装依赖：
```bash
pip install volcengine
```

3. 重启 ComfyUI

## 节点说明

### VLMConfigNode

填入认证信息和项目参数，输出 config 字典。

**参数**：
- `model_name`: 模型名称，默认 `Doubao-seed-1-8`
- `access_key`: Access Key
- `secret_key`: Secret Key
- `account_id`: Account ID
- `project_name`: 项目名称
- `collection_name`: Collection 名称
- `model_version`: 模型版本，默认 `251228`
- `temperature`: Temperature
- `max_tokens`: 最大 Token 数
- `base_url`: API 地址，默认 `http://multi-cloud.byted.org`
- `image_query`: 图片 URL（可选）

### RAGChatNode

执行知识库检索和 LLM 生成。

**输入**：
- `vlm_config`: VLMConfigNode 输出的配置
- `prompt_template`: PE 模板内容，包含 `{{ .user_input }}` 和 `{{ .retrieved_chunks }}` 占位符
- `query`: 用户输入或检索结构化 PE 的输出

**输出**：
- LLM 回复文本

## 使用流程

1. 添加 **VLMConfigNode**，填入所有认证参数
2. 添加 **RAGChatNode**，连接配置
3. 在 `prompt_template` 中填入 PE 模板内容
4. 在 `query` 中填入用户问题
5. 运行获取结果
