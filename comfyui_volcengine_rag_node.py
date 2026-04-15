"""
Volcengine RAG Node for ComfyUI
基于字节跳动知识库API的RAG检索和LLM生成节点

使用方法：
1. 设置VLM配置节点(VLMConfigNode)，填入认证和项目参数
2. 设置RAG主节点(RAGChatNode)，连接配置，输入prompt_template和query
3. 运行获取LLM回复
"""

import json
import requests
from typing import Dict, Any, Optional, List

# ============================================================
# VLM 配置节点 (VLMConfigNode)
# 生成包含所有认证和项目参数的config字典
# ============================================================

class VLMConfigNode:
    """
    VLM配置节点 - 填入认证信息和项目参数
    输出config字典，连接到RAG主节点的vlm_config接口
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "Doubao-seed-1-8", "label": "模型名称"}),
                "access_key": ("STRING", {"default": "", "label": "Access Key"}),
                "secret_key": ("STRING", {"default": "", "label": "Secret Key"}),
                "account_id": ("STRING", {"default": "", "label": "Account ID"}),
                "project_name": ("STRING", {"default": "", "label": "项目名称"}),
                "collection_name": ("STRING", {"default": "", "label": "Collection名称"}),
                "model_version": ("STRING", {"default": "251228", "label": "模型版本"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "label": "Temperature"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "label": "最大Token数"}),
                "base_url": ("STRING", {"default": "https://api-knowledgebase.mlp.cn-beijing.volces.com", "label": "API地址"}),
            },
            "optional": {
                "image_query": ("STRING", {"default": "", "label": "图片URL(可选)"}),
            }
        }

    RETURN_TYPES = ("VLM_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "VLM/RAG"

    def build_config(
        self,
        model_name: str,
        access_key: str,
        secret_key: str,
        account_id: str,
        project_name: str,
        collection_name: str,
        model_version: str,
        temperature: float,
        max_tokens: int,
        base_url: str,
        image_query: str = "",
    ) -> tuple:
        config = {
            "model_name": model_name,
            "access_key": access_key,
            "secret_key": secret_key,
            "account_id": account_id,
            "project_name": project_name,
            "collection_name": collection_name,
            "model_version": model_version,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": base_url,
            "image_query": image_query,
        }
        return (config,)


# ============================================================
# 内部工具函数
# ============================================================

def prepare_request(method: str, path: str, config: Dict, params: Dict = None, data: Dict = None) -> Any:
    """准备签名请求"""
    from volcengine.base.Request import Request
    from volcengine.Credentials import Credentials
    from volcengine.auth.SignerV4 import SignerV4

    if params:
        for key in params:
            if isinstance(params[key], (int, float)):
                params[key] = str(params[key])
            elif isinstance(params[key], list):
                params[key] = ",".join(params[key])

    r = Request()
    r.set_shema("http")
    r.set_method(method)
    r.set_connection_timeout(10)
    r.set_socket_timeout(60)  # 增加到60秒，避免超时
    mheaders = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Host": config.get("base_url", "multi-cloud.byted.org").replace("http://", "").replace("https://", ""),
    }
    r.set_headers(mheaders)
    if params:
        r.set_query(params)
    r.set_host(config.get("base_url", "multi-cloud.byted.org").replace("http://", "").replace("https://", ""))
    r.set_path(path)
    if data is not None:
        r.set_body(json.dumps(data))

    credentials = Credentials(
        config["access_key"],
        config["secret_key"],
        "air",
        "cn-north-1"
    )
    SignerV4.sign(r, credentials)
    return r


def search_knowledge(query: str, image_query: str, config: Dict) -> str:
    """执行知识库检索"""
    method = "POST"
    path = "/api/knowledge/collection/search_knowledge"

    request_params = {
        "project": config["project_name"],
        "name": config["collection_name"],
        "query": query,
        "image_query": image_query if image_query else "",
        "limit": 5,
        "pre_processing": {
            "need_instruction": True,
            "return_token_usage": True,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""}
            ],
            "rewrite": False
        },
        "post_processing": {
            "get_attachment_link": True,
            "rerank_only_chunk": False,
            "rerank_switch": False,
            "chunk_group": True,
            "rerank_model": "doubao-seed-rerank",
            "enable_rerank_threshold": False,
            "retrieve_count": 25,
            "chunk_diffusion_count": 0
        },
        "dense_weight": 0.5
    }

    info_req = prepare_request(method, path, config, data=request_params)
    rsp = requests.request(
        method=info_req.method,
        url="{}{}".format(config["base_url"], info_req.path),
        headers=info_req.headers,
        data=info_req.body
    )
    return rsp.text


def chat_completion(messages: List[Dict], config: Dict) -> str:
    """调用LLM生成"""
    method = "POST"
    path = "/api/knowledge/chat/completions"

    request_params = {
        "messages": messages,
        "stream": False,
        "return_token_usage": True,
        "model": config["model_name"],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "model_version": config["model_version"],
        "thinking": {"type": "enabled"}
    }

    info_req = prepare_request(method, path, config, data=request_params)
    rsp = requests.request(
        method=info_req.method,
        url="{}{}".format(config["base_url"], info_req.path),
        headers=info_req.headers,
        data=info_req.body
    )
    rsp.encoding = "utf-8"
    return rsp.text


def is_vision_model(model_name: str, model_version: str) -> bool:
    """判断是否是VLM模型"""
    MIX_MODEL = ['Doubao-1-5-thinking-pro']
    if not model_name:
        return False
    return (
        "vision" in model_name.lower() or
        "seed" in model_name.lower() or
        (model_name in MIX_MODEL and model_version is not None and model_version.startswith("m"))
    )


def get_content_for_prompt(point: Dict) -> str:
    """从检索结果中提取内容"""
    content = point.get("content", "")
    original_question = point.get("original_question")
    if original_question:
        return "当询问到相似问题时，请参考对应答案进行回答：问题：\"{question}\"。答案：\"{answer}\"".format(
            question=original_question, answer=content)
    return content


def generate_prompt(rsp_txt: str, base_prompt: str, config: Dict, user_query: str) -> List[Dict]:
    """生成发送给LLM的完整prompt"""
    rsp = json.loads(rsp_txt)
    if rsp.get("code") != 0:
        return [{"role": "system", "content": "检索失败：" + rsp.get("message", "")}]

    prompt = ""
    points = rsp.get("data", {}).get("result_list", [])
    using_vlm = is_vision_model(config["model_name"], config["model_version"])
    content = []

    for point in points:
        doc_text_part = ""
        doc_info = point.get("doc_info", {})

        # 提取系统字段
        for system_field in ["point_id", "doc_name", "title"]:
            if system_field in doc_info:
                doc_text_part += f"{system_field}: {doc_info[system_field]}\n"
            elif system_field in point:
                if system_field == "content":
                    doc_text_part += f"content: {get_content_for_prompt(point)}\n"
                elif system_field == "point_id":
                    doc_text_part += f"point_id: \"{point['point_id']}\""
                else:
                    doc_text_part += f"{system_field}: {point[system_field]}\n"

        # 提取table_chunk_fields
        if "table_chunk_fields" in point:
            for self_field in ["template_type", "motion_template", "subject_placeholder"]:
                find_one = next(
                    (item for item in point["table_chunk_fields"] if item["field_name"] == self_field),
                    None
                )
                if find_one:
                    doc_text_part += f"{self_field}: {find_one['field_value']}\n"

        # 提取图片链接
        image_link = None
        if using_vlm and "chunk_attachment" in point and point["chunk_attachment"]:
            image_link = point["chunk_attachment"][0].get("link")

        content.append({'type': 'text', 'text': doc_text_part})
        if image_link:
            content.append({'type': 'image_url', 'image_url': {'url': image_link}})
            prompt += "图片: \n"

        prompt += f"{doc_text_part}\n"

    # 组合完整的system prompt
    # 关键：用实际的user_query替换{{ .user_input }}占位符
    full_system_prompt = base_prompt.replace("{{ .user_input }}", user_query)

    if using_vlm:
        # VLM模式：分割prompt，插入知识库内容
        prompt_parts = full_system_prompt.split("{{ .retrieved_chunks }}")
        if len(prompt_parts) == 2:
            content_pre = {'type': 'text', 'text': prompt_parts[0]}
            content_sub = {'type': 'text', 'text': prompt_parts[1]}
            return [content_pre] + content + [content_sub]
        else:
            return [{"role": "system", "content": full_system_prompt + "\n\n" + prompt}]
    else:
        # LLM模式：直接替换
        full_prompt = full_system_prompt.replace("{{ .retrieved_chunks }}", prompt)
        return [{"role": "system", "content": full_prompt}]


# ============================================================
# RAG 主节点 (RAGChatNode)
# ============================================================

class RAGChatNode:
    """
    RAG对话节点 - 执行知识库检索和LLM生成
    输入：vlm_config（配置）、prompt_template（PE模板）、query（用户问题）
    输出：LLM回复文本
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_config": ("VLM_CONFIG", {"label": "VLM配置"}),
                "prompt_template": ("STRING", {"default": "", "multiline": True, "label": "Prompt模板"}),
                "query": ("STRING", {"default": "", "multiline": True, "label": "Query/用户输入"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "VLM/RAG"

    def run(self, vlm_config: Dict, prompt_template: str, query: str) -> tuple:
        """
        执行RAG检索和LLM生成

        参数:
            vlm_config: VLMConfigNode输出的配置字典
            prompt_template: PE模板内容，包含{{ .user_input }}和{{ .retrieved_chunks }}占位符
            query: 用户输入或检索结构化PE的输出
        """
        # 1. 检索知识库
        search_rsp = search_knowledge(query, vlm_config.get("image_query", ""), vlm_config)

        # 2. 生成prompt
        messages = generate_prompt(search_rsp, prompt_template, vlm_config, query)

        # 3. 添加user message
        user_content = query
        messages.append({"role": "user", "content": user_content})

        # 4. 调用LLM
        chat_rsp = chat_completion(messages, vlm_config)

        # 5. 解析LLM回复
        try:
            rsp_json = json.loads(chat_rsp)
            if rsp_json.get("code") == 0:
                choices = rsp_json.get("data", {}).get("choices", [])
                if choices:
                    result = choices[0].get("message", {}).get("content", "")
                    return (result,)
        except Exception as e:
            return (f"解析错误: {str(e)}\n原始响应: {chat_rsp}",)

        return (f"API返回错误: {chat_rsp[:500]}",)


# ============================================================
# 节点注册
# ============================================================

NODE_CLASS_MAPPINGS = {
    "VLMConfigNode": VLMConfigNode,
    "RAGChatNode": RAGChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMConfigNode": "VLM 配置节点",
    "RAGChatNode": "RAG 对话节点",
}
