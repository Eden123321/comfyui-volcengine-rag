"""
Volcengine RAG Chat Node for ComfyUI
基于字节跳动知识库service/chat API的简化RAG对话节点
"""

import json
import requests
from typing import Dict, Any

# ============================================================
# 配置节点
# ============================================================

class VLMConfigNode:
    """VLM配置节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apikey": ("STRING", {"default": "", "label": "API Key"}),
                "service_resource_id": ("STRING", {"default": "kb-service-b2a3d93f6a9f93ed", "label": "Service Resource ID"}),
                "base_url": ("STRING", {"default": "api-knowledgebase.mlp.cn-beijing.volces.com", "label": "API地址"}),
            }
        }

    RETURN_TYPES = ("VLM_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "VLM/RAG"

    def build_config(
        self,
        apikey: str,
        service_resource_id: str,
        base_url: str,
    ) -> tuple:
        config = {
            "apikey": apikey,
            "service_resource_id": service_resource_id,
            "base_url": base_url,
        }
        return (config,)


# ============================================================
# 主节点
# ============================================================

class RAGChatNode:
    """RAG对话节点 - 调用service/chat接口"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_config": ("VLM_CONFIG", {"label": "VLM配置"}),
                "query": ("STRING", {"default": "", "multiline": True, "label": "Query/用户输入"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "VLM/RAG"

    def run(self, vlm_config: Dict, query: str) -> tuple:
        """调用service/chat接口"""
        apikey = vlm_config.get("apikey", "")
        service_resource_id = vlm_config.get("service_resource_id", "")
        base_url = vlm_config.get("base_url", "").replace("http://", "").replace("https://", "")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json;charset=UTF-8",
            "Host": base_url,
            "Authorization": f"Bearer {apikey}"
        }

        request_params = {
            "service_resource_id": service_resource_id,
            "messages": [{"role": "user", "content": query}],
            "stream": False
        }

        full_url = f"https://{base_url}/api/knowledge/service/chat"

        try:
            rsp = requests.post(
                full_url,
                headers=headers,
                json=request_params,
                timeout=60
            )
            rsp.encoding = "utf-8"

            if rsp.status_code != 200:
                return (f"HTTP错误: {rsp.status_code}\n{rsp.text[:500]}",)

            result = json.loads(rsp.text)

            if result.get("code") != 0:
                return (f"API错误: {result.get('message', '未知错误')}",)

            # 提取回复内容
            data = result.get("data", {})
            generated_answer = data.get("generated_answer", "")
            if generated_answer:
                return (generated_answer,)

            return (f"无法解析响应: {rsp.text[:500]}",)

        except Exception as e:
            return (f"请求异常: {str(e)}",)


# ============================================================
# 节点注册
# ============================================================

NODE_CLASS_MAPPINGS = {
    "VLMConfigNode": VLMConfigNode,
    "RAGChatNode": RAGChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMConfigNode": "VLM Config Node",
    "RAGChatNode": "RAG Chat Node",
}
