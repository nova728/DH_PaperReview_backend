import os
from dataclasses import dataclass

@dataclass
class VllmConfig:
    # Automatic Review 模型配置
    automatic_review_url: str = "http://127.0.0.1:8000"
    automatic_review_model: str = "scientific-reviewer-7b"
    
    # Deep Review 模型配置
    deep_review_url: str = "http://127.0.0.1:8001"
    deep_review_model: str = "deep-review-7b"
    
    timeout: int = 300
    max_context_length: int = 64000
    batch_size: int = 1
    max_parallel_requests: int = 1

class AppConfig:
    def __init__(self):
        self.vllm = VllmConfig(
            automatic_review_url=os.getenv('AUTOMATIC_REVIEW_URL', 'http://127.0.0.1:8000'),
            automatic_review_model=os.getenv('AUTOMATIC_REVIEW_MODEL', 'scientific-reviewer-7b'),
            deep_review_url=os.getenv('DEEP_REVIEW_URL', 'http://127.0.0.1:8001'),
            deep_review_model=os.getenv('DEEP_REVIEW_MODEL', 'deep-review-7b'),
            timeout=int(os.getenv('VLLM_TIMEOUT', '300'))
        )
        
        self.vllm.base_url = self.vllm.automatic_review_url
        self.vllm.model_name = self.vllm.automatic_review_model