import os
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMIntegration:
    
    def __init__(self):
        self.model_registry = {
            # OpenAI series
            "GPT-4": {
                "provider": "openai",
                "config": {
                    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "gpt-4"
                }
            },
            
            # ali series
            "Qwen2.5-72b": {
                "provider": "dashscope",
                "config": {
                    "api_key": os.getenv("DASHSCOPE_API_KEY"),
                    "model": "qwen1.5-72b-chat"
                }
            },
            "Qwen-max": {
                "provider": "dashscope",
                "config": {
                    "api_key": os.getenv("DASHSCOPE_API_KEY"),
                    "model": "qwen-max"
                }
            },
            
            # Meta series
            "llama3-8b": {
                "provider": "ollama",
                "config": {"model": "llama3:8b"}
            },
            "llama3-70b": {
                "provider": "ollama",
                "config": {"model": "llama3:70b"}
            },
            
            # zhipuAI
            "GLM-4": {
                "provider": "zhipuai",
                "config": {
                    "api_key": os.getenv("ZHIPUAI_API_KEY"),
                    "model": "glm-4"
                }
            },
            
            # DeepSeek
            "deepseek-r1:8b": {
                "provider": "openai",
                "config": {
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": os.getenv("DEEPSEEK_API_KEY"),
                    "model": "deepseek-r1"
                }
            },
            "deepseek-r1:70b": {
                "provider": "openai",
                "config": {
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": os.getenv("DEEPSEEK_API_KEY"),
                    "model": "deepseek-r1"
                }
            }
        }

    def get_response(self, system_prompt: str, user_prompt: str, model_name: str) -> Optional[str]:
        """统一调用入口"""
        if model_name not in self.model_registry:
            logger.error(f"Model {model_name} not registered")
            return None
            
        model_info = self.model_registry[model_name]
        try:
            if model_info["provider"] == "openai":
                return self._call_openai(system_prompt, user_prompt, model_info["config"])
            elif model_info["provider"] == "dashscope":
                return self._call_dashscope(system_prompt, user_prompt, model_info["config"])
            elif model_info["provider"] == "ollama":
                return self._call_ollama(system_prompt, user_prompt, model_info["config"])
            elif model_info["provider"] == "zhipuai":
                return self._call_zhipuai(system_prompt, user_prompt, model_info["config"])
        except Exception as e:
            logger.error(f"Error calling {model_name}: {str(e)}")
            return None

    def _call_openai(self, system_prompt: str, user_prompt: str, config: dict) -> str:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url", "https://api.openai.com/v1")
        )
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    def _call_dashscope(self, system_prompt: str, user_prompt: str, config: dict) -> str:
        import dashscope
        
        dashscope.api_key = config["api_key"]
        response = dashscope.Generation.call(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.output.choices[0].message.content

    def _call_ollama(self, system_prompt: str, user_prompt: str, config: dict) -> str:
        from ollama import chat
        
        response = chat(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response["message"]["content"]

    def _call_zhipuai(self, system_prompt: str, user_prompt: str, config: dict) -> str:
        from zhipuai import ZhipuAI
        
        client = ZhipuAI(api_key=config["api_key"])
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
