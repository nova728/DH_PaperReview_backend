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
        """调用LLM生成评审（使用默认模型）"""
        return self._call_llm_for_review_with_model(prompt, temperature, max_tokens, model_name=None)
    
    def _call_llm_for_review_with_model(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8192, model_name: str = None) -> str:
        """调用LLM生成评审（可指定模型）"""
        if self.vllm_service:
            try:
                # 使用VllmService的通用文本生成方法
                result = self.vllm_service.generate_text(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model_name=model_name
                )
                
                logger.info(f"生成的评审长度: {len(result):,} 字符")
                return result
            except Exception as e:
                logger.error(f"调用VllmService失败: {str(e)}")
                return f"Error generating review: {str(e)}"
        else:
            # 如果没有VllmService，返回占位符
            return "This is a placeholder review content. Please provide VllmService for actual LLM call."
    
    def generate_deep_review(self, paper_content: str, temperature: float = 0.0, max_tokens: int = 8192) -> Dict[str, Any]:
        """
        生成深度评审 - 使用 deep review-7b 模型
        
        Args:
            paper_content: 论文内容
            temperature: 温度参数，控制输出的随机性，默认0.0表示确定性输出
            max_tokens: 最大生成token数量，默认8192
            
        Returns:
            包含评审结果的字典
        """
        try:
            # Deep review prompt
            deep_review_prompt = (
                "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Fast Mode.\n\n"
                "Strictly follow the instructions below:\n"
                "1. Read the paper content between <paper>...</paper>.\n"
                "2. Return your answer using EXACTLY the following four sections in this order.\n"
                "3. Use plain text only. Do not add extra sections, headers, bullets, JSON, or braces.\n\n"
                "Summary:\n"
                "<Provide a concise paragraph summarizing the work.>\n\n"
                "Strengths:\n"
                "<Provide one concise paragraph describing the main strengths.>\n\n"
                "Weaknesses:\n"
                "<Provide one concise paragraph describing the main weaknesses or concerns.>\n\n"
                "Decision:\n"
                "<Provide a single-word recommendation such as Accept, Weak Accept, Borderline, Weak Reject, or Reject.>\n\n"
                "Content of the paper to be reviewed:\n"
                "<paper>\n"
                f"{paper_content}\n"
                "</paper>"
            )
            review_content = self._call_llm_for_review_with_model(
                prompt=deep_review_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model_name="deep-review-7b"
            )
            
            return {
                "type": "deep_review",
                "content": review_content,
                "source": "deep-review-7b"
            }
            
        except Exception as e:
            logger.error(f"生成深度评审失败: {str(e)}")
            return {"error": str(e)}
    
    def format_deep_review_to_frontend(self, review_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        将 deep review 结果格式化为前端期望的格式
        基于 deep review 的 11 个返回部分进行格式化
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
                content = review_result.get("content", "")
            
            if not content:
                return [{
                    "name": "Error", 
                    "content": "评审内容为空"
                }]
            
            # 解析评审内容的结构化部分
            sections = self._parse_deep_review_sections(content)
            
            reviews = []
            
            # 按照指定的顺序添加各个部分
            section_names = [
                "summary", "soundness", "presentation", "contribution",
                "strengths", "weaknesses", "suggestions", "questions",
                "rating", "confidence", "decision"
            ]
            
            section_display_names = {
                "summary": "Summary",
                "soundness": "Soundness",
                "presentation": "Presentation",
                "contribution": "Contribution",
                "strengths": "Strengths",
                "weaknesses": "Weaknesses",
                "suggestions": "Suggestions",
                "questions": "Questions",
                "rating": "Rating",
                "confidence": "Confidence",
                "decision": "Decision"
            }
            
            for section_name in section_names:
                content_text = sections.get(section_name, f"{section_display_names[section_name]}信息未找到")
                reviews.append({
                    "name": section_display_names[section_name],
                    "content": content_text
                })
            
            return reviews
            
        except Exception as e:
            logger.error(f"格式化深度评审结果失败: {str(e)}")
            return [{
                "name": "Error",
                "content": f"格式化失败: {str(e)}"
            }]
    
    def _parse_deep_review_sections(self, content: str) -> Dict[str, str]:
        """
        解析深度评审内容的结构化部分
        支持 Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses,
        Suggestions, Questions, Rating, Confidence, Decision
        """
        sections = {}
        
        # 定义各部分的标识符
        section_patterns = {
            "summary": [r"\*\*Summary:\*\*", r"# Summary", r"##\s*Summary", r"Summary:", r"SUMMARY"],
            "soundness": [r"\*\*Soundness:\*\*", r"# Soundness", r"##\s*Soundness", r"Soundness:", r"SOUNDNESS"],
            "presentation": [r"\*\*Presentation:\*\*", r"# Presentation", r"##\s*Presentation", r"Presentation:", r"PRESENTATION"],
            "contribution": [r"\*\*Contribution:\*\*", r"# Contribution", r"##\s*Contribution", r"Contribution:", r"CONTRIBUTION"],
            "strengths": [r"\*\*Strengths:\*\*", r"# Strengths", r"##\s*Strengths", r"Strengths:", r"STRENGTHS"],
            "weaknesses": [r"\*\*Weaknesses:\*\*", r"# Weaknesses", r"##\s*Weaknesses", r"Weaknesses:", r"WEAKNESSES"],
            "suggestions": [r"\*\*Suggestions:\*\*", r"# Suggestions", r"##\s*Suggestions", r"Suggestions:", r"SUGGESTIONS"],
            "questions": [r"\*\*Questions:\*\*", r"# Questions", r"##\s*Questions", r"Questions:", r"QUESTIONS"],
            "rating": [r"\*\*Rating:\*\*", r"# Rating", r"##\s*Rating", r"Rating:", r"RATING"],
            "confidence": [r"\*\*Confidence:\*\*", r"# Confidence", r"##\s*Confidence", r"Confidence:", r"CONFIDENCE"],
            "decision": [r"\*\*Decision:\*\*", r"# Decision", r"##\s*Decision", r"Decision:", r"DECISION"]
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