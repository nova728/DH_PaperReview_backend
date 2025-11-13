import os
from dataclasses import dataclass

@dataclass
class VllmConfig:
    # Automatic Review 模型配置
    automatic_review_url: str = "http://127.0.0.1:8011"
    automatic_review_model: str = "scientific-reviewer-7b"
    
    # Deep Review 模型配置
    deep_review_url: str = "http://127.0.0.1:8012"
    deep_review_model: str = "deep-review-7b"
    
    # 通用配置
    timeout: int = 300
    max_context_length: int = 64000
    batch_size: int = 1
    max_parallel_requests: int = 1

class AppConfig:
    def __init__(self):
        automatic_review_url = (os.getenv('AUTOMATIC_REVIEW_URL') or 'http://127.0.0.1:8011').strip()
        automatic_review_model = (os.getenv('AUTOMATIC_REVIEW_MODEL') or 'scientific-reviewer-7b').strip()
        
        deep_review_url_env = (os.getenv('DEEP_REVIEW_URL') or '').strip()
        deep_review_model_env = (os.getenv('DEEP_REVIEW_MODEL') or '').strip()
        
        deep_review_url = deep_review_url_env or 'http://127.0.0.1:8012'
        deep_review_model = deep_review_model_env or 'deep-review-7b'
        
        self.vllm = VllmConfig(
            automatic_review_url=automatic_review_url,
            automatic_review_model=automatic_review_model,
            deep_review_url=deep_review_url,
            deep_review_model=deep_review_model,
            timeout=int(os.getenv('VLLM_TIMEOUT', '300'))
        )