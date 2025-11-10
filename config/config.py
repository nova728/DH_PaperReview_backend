import os
from dataclasses import dataclass

@dataclass
class VllmConfig:
    base_url: str = "http://127.0.0.1:8000"  
    model_name: str = "scientific-reviewer-7b"  
    timeout: int = 300
    max_context_length: int = 64000  # 32k上下文
    batch_size: int = 1
    max_parallel_requests: int = 1

class AppConfig:
    def __init__(self):
        self.vllm = VllmConfig(
            base_url=os.getenv('VLLM_BASE_URL', 'http://127.0.0.1:8000'),
            model_name=os.getenv('VLLM_MODEL_NAME', 'scientific-reviewer-7b'),
            timeout=int(os.getenv('VLLM_TIMEOUT', '300'))
        )