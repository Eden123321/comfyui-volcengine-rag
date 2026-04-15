# Volcengine RAG Node for ComfyUI

ComfyUI custom node for RAG-based knowledge base retrieval and LLM generation.

## Installation

1. Copy `comfyui_volcengine_rag_node.py` to ComfyUI's `custom_nodes` folder

2. Install dependency:
```bash
pip install volcengine
```

3. Restart ComfyUI

## Nodes

### VLMConfigNode

Configure API credentials and project parameters. Outputs a config dictionary.

**Parameters**:
- `model_name`: Model name, default `Doubao-seed-1-8`
- `access_key`: Access Key
- `secret_key`: Secret Key
- `account_id`: Account ID
- `project_name`: Project name
- `collection_name`: Collection name
- `model_version`: Model version, default `251228`
- `temperature`: Temperature
- `max_tokens`: Max tokens
- `base_url`: API base URL
- `image_query`: Image URL (optional)

### RAGChatNode

Perform knowledge base retrieval and LLM generation.

**Inputs**:
- `vlm_config`: Output from VLMConfigNode
- `prompt_template`: PE template with `{{ .user_input }}` and `{{ .retrieved_chunks }}` placeholders
- `query`: User input or structured data

**Output**:
- LLM response text

## Usage

1. Add **VLMConfigNode** and fill in credentials
2. Add **RAGChatNode** and connect the config
3. Fill in PE template in `prompt_template`
4. Fill in user query in `query`
5. Run to get results
