#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Review Service - 集成Automatic_Review项目的功能
"""

import json
import logging
import os
import sys
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
automatic_review_path = Path(__file__).parent.parent.parent / "Automatic_Review"
if automatic_review_path.exists():
    sys.path.append(str(automatic_review_path))

logger = logging.getLogger(__name__)

class AutomaticReviewService:
    """自动评审服务 - 集成Automatic_Review项目的功能"""
    
    def __init__(self, config, vllm_service=None):
        self.config = config
        self.vllm_service = vllm_service
        self.automatic_review_path = automatic_review_path
        self.evaluation_path = automatic_review_path / "evaluation"
        self.generation_path = automatic_review_path / "generation"
        
        # 检查Automatic_Review项目是否存在
        if not automatic_review_path.exists():
            logger.warning("Automatic_Review项目不存在，某些功能可能不可用")
    
    def generate_review(self, paper_content: str, temperature: float = 0.0, max_tokens: int = 8192) -> Dict[str, Any]:
        """
        生成评审 - 使用Automatic_Review的原始功能
        
        Args:
            paper_content: 论文内容
            temperature: 温度参数，控制输出的随机性，默认0.0表示确定性输出
            max_tokens: 最大生成token数量，默认8192
            
        Returns:
            包含评审结果的字典
        """
        try:
            return self._generate_review_using_automatic_review(paper_content, temperature, max_tokens)
        except Exception as e:
            logger.error(f"生成评审失败: {str(e)}")
            return {"error": str(e)}
    
    def format_automatic_review_to_frontend(self, review_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        将automatic_review结果格式化为前端期望的格式
        基于prompt_generate_review_v2.txt的4个返回部分进行格式化
        """
        try:
            if "error" in review_result:
                return [{
                    "name": "Error",
                    "content": f"评审生成失败: {review_result.get('error', '未知错误')}"
                }]
            
            # 获取评审内容
            content = review_result.get("result", {}).get("content", "")
            if not content:
                return [{
                    "name": "Error", 
                    "content": "评审内容为空"
                }]
            
            # 解析评审内容的4个结构化部分
            sections = self._parse_review_sections(content)
            
            reviews = []
            
            # 1. Summary 
            summary_content = sections.get("summary", "论文概述信息未找到")
            reviews.append({
                "name": "Summary",
                "content": f"{summary_content}"
            })
            
            # 2. Strengths 
            strengths_content = sections.get("strengths", "论文优势信息未找到")
            reviews.append({
                "name": "Strengths", 
                "content": f"{strengths_content}"
            })
            
            # 3. Weaknesses 
            weaknesses_content = sections.get("weaknesses", "论文不足信息未找到")
            reviews.append({
                "name": "Weaknesses",
                "content": f"{weaknesses_content}"
            })
            
            # 4. Decision 
            decision_content = sections.get("decision", "评审决策信息未找到")
            if not decision_content or decision_content == "评审决策信息未找到":
                decision_from_result = review_result.get("result", {}).get("Decision", "")
                if decision_from_result:
                    decision_content = f"最终决策：{decision_from_result}"
            
            reviews.append({
                "name": "Decision",
                "content": f"{decision_content}"
            })
            
            return reviews
            
        except Exception as e:
            logger.error(f"格式化评审结果失败: {str(e)}")
            return [{
                "name": "Error",
                "content": f"格式化失败: {str(e)}"
            }]

    def _parse_review_sections(self, content: str) -> Dict[str, str]:
        """
        解析评审内容的结构化部分
        基于prompt_generate_review_v2.txt的返回格式：Summary, Strengths, Weaknesses, Decision
        """
        sections = {}
        
        # 定义各部分的标识符
        section_patterns = {
            "summary": [r"\*\*Summary:\*\*", r"# Summary", r"##\s*Summary", r"Summary:", r"SUMMARY"],
            "strengths": [r"\*\*Strengths:\*\*", r"# Strengths", r"##\s*Strengths", r"Strengths:", r"STRENGTHS"],
            "weaknesses": [r"\*\*Weaknesses:\*\*", r"# Weaknesses", r"##\s*Weaknesses", r"Weaknesses:", r"WEAKNESSES"],
            "decision": [r"\*\*Decision\*\*", r"# Decision", r"##\s*Decision", r"Decision:", r"DECISION"]
        }
        
        # 按行分割内容
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            section_found = False
            for section_name, patterns in section_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        if current_section and current_content:
                            sections[current_section] = '\n'.join(current_content).strip()
                        
                        current_section = section_name
                        current_content = []
                        section_found = True
                        break
                if section_found:
                    break
            
            if not section_found and current_section:
                current_content.append(line)
            elif not section_found and not current_section:
                current_content.append(line)
        
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        elif current_content and not sections:
            sections["summary"] = '\n'.join(current_content).strip()
        
        return sections

    def _extract_relevant_sentences(self, text: str, keywords: List[str]) -> List[str]:
        """从文本中提取包含关键词的相关句子"""
        if not text:
            return []
        
        sentences = []
        raw_sentences = re.split(r'[.!?]+', text)
        
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_lower = sentence.lower()
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                sentences.append(sentence)
        
        return sentences[:3]  # 最多返回3个相关句子

    def _generate_review_using_automatic_review(self, paper_content: str, temperature: float = 0.0, max_tokens: int = 8192) -> Dict[str, Any]:
        """使用Automatic_Review的原始功能生成评审"""
        prompt_template = self._load_prompt_template("generation", "prompt_generate_review_v2.txt")     
        if prompt_template:
            template_parts = prompt_template.split("<paper>")
        
        # 在<paper>标签处插入论文内容
        if prompt_template and "<paper>" in prompt_template:
            prompt = prompt_template.replace("<paper>", "") + paper_content + "\n</paper>"
        else:
            prompt = (prompt_template or "") + "\n<paper>\n" + paper_content + "\n</paper>"
        
        review_content = self._call_llm_for_review(prompt, temperature, max_tokens)
        
        return {
            "type": "automatic_review",
            "content": review_content,
            "source": "Automatic_Review"
        }
    
    def _load_prompt_template(self, module: str, filename: str) -> Optional[str]:
        """加载提示词模板"""
        try:
            if module == "generation":
                prompt_path = self.generation_path / "prompts" / filename
            elif module == "evaluation":
                prompt_path = self.evaluation_path / "prompts" / filename
            else:
                return None
            
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"提示词模板文件不存在: {prompt_path}")
                return None
                
        except Exception as e:
            logger.error(f"加载提示词模板失败: {str(e)}")
            return None
    
    def _call_llm_for_review(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192) -> str:
        """调用LLM生成评审"""
        if self.vllm_service:
            try:
                # 使用VllmService的通用文本生成方法
                result = self.vllm_service.generate_text(
                    prompt=prompt,
                    temperature=temperature,  # 使用传入的温度参数
                    max_tokens=max_tokens     # 使用传入的最大生成token参数
                )
                
                logger.info(f"生成的评审长度: {len(result):,} 字符")
                return result
            except Exception as e:
                logger.error(f"调用VllmService失败: {str(e)}")
                return f"Error generating review: {str(e)}"
        else:
            # 如果没有VllmService，返回占位符
            return "This is a placeholder review content. Please provide VllmService for actual LLM call."

