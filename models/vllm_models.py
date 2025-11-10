from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class VllmMessage:
    role: str
    content: str

@dataclass
class VllmRequest:
    model: str
    messages: List[VllmMessage]
    max_tokens: int = 8192 
    temperature: float = 0.0  # 确定性输出
    stream: bool = False  # 添加流式输出支持
    
    def to_dict(self):
        return {
            'model': self.model,
            'messages': [{'role': msg.role, 'content': msg.content} for msg in self.messages],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'stream': self.stream
        }

@dataclass
class VllmResponse:
    choices: List[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(choices=data.get('choices', []))
    
    def get_content(self) -> str:
        if self.choices and len(self.choices) > 0:
            return self.choices[0].get('message', {}).get('content', '')
        return ''
