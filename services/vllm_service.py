import requests
import logging
import json
from typing import Optional, Generator
from config.config import AppConfig
from models.vllm_models import VllmRequest, VllmMessage, VllmResponse

logger = logging.getLogger(__name__)

class VllmService:
    def __init__(self, config: AppConfig):
        self.config = config
        
        # 支持两种配置方式
        # 方案1：单个端点多模型
        if hasattr(config.vllm, 'base_url'):
            self.base_url = config.vllm.base_url.rstrip('/')
            self.model_name = config.vllm.model_name
            self.single_endpoint = True
            logger.info(f"使用单端点模式: {self.base_url}")
        # 方案2：多个端点
        else:
            self.automatic_review_url = config.vllm.automatic_review_url.rstrip('/')
            self.deep_review_url = config.vllm.deep_review_url.rstrip('/')
            self.automatic_review_model = config.vllm.automatic_review_model
            self.deep_review_model = config.vllm.deep_review_model
            self.single_endpoint = False
            logger.info(f"使用多端点模式: Automatic={self.automatic_review_url}, Deep={self.deep_review_url}")
        
        self._warmup_model()
    
    def _get_endpoint_and_model(self, model_name: str = None):
        """根据模型名称获取对应的端点和模型名"""
        if self.single_endpoint:
            # 单端点模式：使用同一个 URL，不同的 model_name
            url = self.base_url
            model = model_name if model_name else self.model_name
        else:
            # 多端点模式：根据模型名称选择不同的 URL
            if model_name == "deep-review-7b":
                url = self.deep_review_url
                model = self.deep_review_model
            else:
                url = self.automatic_review_url
                model = self.automatic_review_model
        
        return url, model
    
    def generate_text(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192, model_name: str = None) -> str:
        """通用文本生成方法，直接接收完整的prompt"""
        logger.info("Calling vLLM to generate text")
        logger.info(f"温度设置: {temperature}, 最大生成token: {max_tokens}")
        
        # 获取对应的端点和模型
        url, model = self._get_endpoint_and_model(model_name)
        logger.info(f"使用模型: {model}, 端点: {url}")
        
        try:
            # 创建请求
            vllm_request = VllmRequest(
                model=model,
                messages=[
                    VllmMessage(role="user", content=prompt)
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 调用API
            logger.info(f"调用 API: {url}/v1/chat/completions, 模型: {model}")
            response = self._call_vllm_api(vllm_request, url)
            content = response.get_content()
            
            if not content.strip():
                raise RuntimeError("vLLM 服务返回空结果")
                
            logger.info(f"vLLM 文本生成完成，输出长度: {len(content)} 字符")
            return content
            
        except Exception as e:
            logger.error(f"vLLM 调用失败: {str(e)}")
            raise RuntimeError(f"文本生成失败: {str(e)}")
    
    def generate_text_stream(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192, model_name: str = None) -> Generator[str, None, None]:
        """通用文本生成方法（流式），直接接收完整的prompt"""
        logger.info("Calling vLLM to generate text (streaming)")
        
        # 获取对应的端点和模型
        url, model = self._get_endpoint_and_model(model_name)
        
        try:
            # 创建流式请求
            vllm_request = VllmRequest(
                model=model,
                messages=[
                    VllmMessage(role="user", content=prompt)
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            # 调用流式API
            for chunk in self._call_vllm_stream_api(vllm_request, url):
                yield chunk
            
            logger.info("vLLM 文本生成流式完成")
            
        except Exception as e:
            logger.error(f"vLLM 流式调用失败: {str(e)}")
            raise RuntimeError(f"文本流式生成失败: {str(e)}")
    
    def _call_vllm_api(self, vllm_request: VllmRequest, url: str = None) -> VllmResponse:
        """调用API"""
        if url is None:
            url = self.base_url if self.single_endpoint else self.automatic_review_url
        
        api_url = f"{url}/v1/chat/completions"
        
        try:
            response = requests.post(
                api_url,
                json=vllm_request.to_dict(),
                timeout=self.config.vllm.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            return VllmResponse.from_dict(response.json())
            
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM API 调用失败: {str(e)}")
            raise RuntimeError(f"vLLM API 调用失败: {str(e)}")

    def _call_vllm_stream_api(self, vllm_request: VllmRequest, url: str = None) -> Generator[str, None, None]:
        """调用流式API"""
        if url is None:
            url = self.base_url if self.single_endpoint else self.automatic_review_url
            
        api_url = f"{url}/v1/chat/completions"
        
        try:
            response = requests.post(
                api_url,
                json=vllm_request.to_dict(),
                timeout=self.config.vllm.timeout,
                headers={'Content-Type': 'application/json'},
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_content = line[6:]  # 移除 'data: ' 前缀
                        
                        if data_content.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data_content)
                            choices = chunk_data.get('choices', [])
                            if choices and len(choices) > 0:
                                delta = choices[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            # 忽略无法解析的行
                            continue
            
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM 流式API 调用失败: {str(e)}")
            raise RuntimeError(f"vLLM 流式API 调用失败: {str(e)}")

    def _warmup_model(self):
        """预热模型"""
        try:
            logger.info("正在预热vLLM模型...")
            
            if self.single_endpoint:
                url = self.base_url
                model = self.model_name
            else:
                url = self.automatic_review_url
                model = self.automatic_review_model
            
            dummy_request = VllmRequest(
                model=model,
                messages=[
                    VllmMessage(role="system", content="You are an AI assistant."),
                    VllmMessage(role="user", content="test")
                ],
                max_tokens=10,
                temperature=0.1
            )
            self._call_vllm_api(dummy_request, url)
            logger.info("vLLM模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败，但服务仍可正常运行: {str(e)}")