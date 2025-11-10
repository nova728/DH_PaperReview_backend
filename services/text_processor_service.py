import logging
from typing import Dict, Any, List, Optional
import re

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False

logger = logging.getLogger(__name__)

class TextProcessorService:
    MAX_LENGTH = 82768  # 32k字符限制
    MAX_TOKENS = 32000  # 32k token限制
    
    def __init__(self, include_authors=False, tokenizer_path=None):
        """
        初始化文本处理服务
        
        Args:
            include_authors (bool): 是否包含作者信息，默认False
                                  对于peer review，建议设为False以避免偏见
            tokenizer_path (str): tokenizer路径，用于token级别处理
        """
        self.include_authors = include_authors
        self.tokenizer = None
        
        # 初始化tokenizer（如果可用）
        if HAS_TOKENIZER and tokenizer_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"成功加载tokenizer: {tokenizer_path}")
            except Exception as e:
                logger.warning(f"无法加载tokenizer {tokenizer_path}: {str(e)}")
                self.tokenizer = None
    
    def process_paper_json(self, paper_json: Dict[str, Any], auto_truncate: bool = True) -> str:
        """
        JSON论文转文本
        
        Args:
            paper_json: 论文JSON数据
            auto_truncate: 是否自动截断到最大长度，默认True
        """
        logger.info("处理JSON格式论文数据")
        logger.info(f"自动截断设置: {auto_truncate}")
        
        try:
            text_parts = []
            
            # 处理标题
            title = self._extract_title(paper_json)
            if title:
                text_parts.append(f"Title: {title}\n")
                logger.info(f"论文标题: {title}")
            
            # 处理作者（可选）
            if self.include_authors:
                authors_text = self._extract_authors(paper_json)
                if authors_text:
                    text_parts.append(f"Authors: {authors_text}\n")
                    logger.info(f"包含作者信息: {len(authors_text)} 字符")
            else:
                # 对于peer review，跳过作者信息以避免偏见
                logger.info("跳过作者信息处理（匿名评审模式）")
            
            # 处理发表信息
            publication_text = self._extract_publication(paper_json)
            if publication_text:
                text_parts.append(f"Publication: {publication_text}\n")
            
            # 处理摘要
            abstract_text = self._extract_abstract(paper_json)
            if abstract_text:
                text_parts.append("Abstract:\n")
                text_parts.append(f"{abstract_text}\n")
            
            # 处理正文
            body_text = self._extract_body(paper_json)
            if body_text:
                text_parts.append("\nMain Content:\n")
                text_parts.append(body_text)
            
            # 处理参考文献
            references_text = self._extract_references(paper_json)
            if references_text:
                text_parts.append("\nReferences:\n")
                text_parts.append(references_text)
            
            # 合并文本
            full_text = "\n".join(text_parts)
            
            # 根据参数决定是否截断
            if auto_truncate:
                truncated_text = self._truncate_to_max_tokens(full_text)
                if truncated_text != full_text:
                    logger.warning(f"论文内容已截断: 原始长度 {len(full_text)} 字符 => 截断后 {len(truncated_text)} 字符")
                    if self.tokenizer:
                        try:
                            orig_tokens = len(self.tokenizer.encode(full_text))
                            trunc_tokens = len(self.tokenizer.encode(truncated_text))
                            logger.warning(f"Token 数量: 原始 {orig_tokens} => 截断后 {trunc_tokens}")
                        except Exception as e:
                            logger.warning(f"计算token数量失败: {str(e)}")
                return truncated_text
            else:
                logger.info(f"未启用自动截断，返回完整论文内容: {len(full_text)} 字符")
                if self.tokenizer:
                    try:
                        tokens = len(self.tokenizer.encode(full_text))
                        logger.info(f"完整论文的token数量: {tokens}")
                        if tokens > self.MAX_TOKENS:
                            logger.warning(f"警告: 论文token数量 ({tokens}) 超过了最大限制 ({self.MAX_TOKENS})，可能会被模型截断")
                    except Exception as e:
                        logger.warning(f"计算token数量失败: {str(e)}")
                return full_text
            
        except Exception as e:
            logger.error(f"处理JSON论文数据失败: {str(e)}")
            raise RuntimeError(f"处理JSON论文数据失败: {str(e)}")
    
    def _truncate_to_max_length(self, text: str) -> str:
        """截断到最大字符长度"""
        if len(text) <= self.MAX_LENGTH:
            logger.info(f"文本字符长度 ({len(text)}) 在限制范围内，不需要截断")
            return text
        
        logger.warning(f"文本字符长度 ({len(text)}) 超过限制 ({self.MAX_LENGTH})，进行截断")
        truncated_text = text[:self.MAX_LENGTH - 100] + "..."
        logger.info(f"截断后字符长度: {len(truncated_text)}")
        return truncated_text
    
    def _truncate_to_max_tokens(self, text: str, max_tokens: int = None) -> str:
        """按token数量截断（与predict.py对齐）"""
        if max_tokens is None:
            max_tokens = self.MAX_TOKENS
            
        # 如果没有tokenizer，回退到字符截断
        if not self.tokenizer:
            logger.warning("没有可用的tokenizer，使用字符截断")
            return self._truncate_to_max_length(text)
        
        try:
            # token级别处理
            tokens = self.tokenizer.encode(text)
            token_count = len(tokens)
            
            if token_count <= max_tokens:
                logger.info(f"文本token数量 ({token_count}) 在限制范围内，不需要截断")
                return text
            
            # 截断并解码
            logger.warning(f"文本token数量 ({token_count}) 超过限制 ({max_tokens})，进行截断")
            truncated_tokens = tokens[:max_tokens - 100]  # 留一些余量给模型生成
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            logger.info(f"截断后token数量: {len(self.tokenizer.encode(truncated_text))}")
            return truncated_text
            
        except Exception as e:
            logger.warning(f"Token截断失败，使用字符截断: {str(e)}")
            return self._truncate_to_max_length(text)

    def _extract_title(self, paper_json: Dict[str, Any]) -> str:
        """提取标题"""
        return paper_json.get('title', '').strip()

    def _extract_authors(self, paper_json: Dict[str, Any]) -> str:
        """提取作者"""
        authors = paper_json.get('author', [])
        if not isinstance(authors, list):
            return ""
        
        author_names = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get('name', '').strip()
                if name:
                    author_names.append(name)
            elif isinstance(author, str):
                author_names.append(author.strip())
        
        return ', '.join(author_names)

    def _extract_publication(self, paper_json: Dict[str, Any]) -> str:
        """提取发表信息"""
        publication = paper_json.get('publication', {})
        if not isinstance(publication, dict):
            return ""
        
        pub_parts = []
        
        # 发表日期
        date = publication.get('date', '').strip()
        if date:
            pub_parts.append(f"Publication Date: {date}")
        
        # 发表者
        publisher = publication.get('publisher', {})
        if isinstance(publisher, dict):
            pub_name = publisher.get('name', '').strip()
            if pub_name:
                pub_parts.append(f"Publisher: {pub_name}")
        
        return ', '.join(pub_parts)
    
    def _extract_abstract(self, paper_json: Dict[str, Any]) -> str:
        """提取摘要"""
        abstract = paper_json.get('abstract', [])
        if not isinstance(abstract, list):
            return ""
        
        abstract_parts = []
        for abstract_item in abstract:
            if isinstance(abstract_item, list) and abstract_item:
                abstract_parts.append(str(abstract_item[0]).strip())
            elif isinstance(abstract_item, str):
                abstract_parts.append(abstract_item.strip())
        
        return '\n'.join(abstract_parts)
    
    def _extract_body(self, paper_json: Dict[str, Any]) -> str:
        """提取正文"""
        body = paper_json.get('body', [])
        if not isinstance(body, list):
            return ""
        
        body_parts = []
        for section in body:
            if not isinstance(section, dict):
                continue
            
            # 章节标题
            section_title = self._extract_section_title(section)
            if section_title:
                body_parts.append(f"\n{section_title}")
            
            # 段落内容
            paragraphs = self._extract_paragraphs(section)
            if paragraphs:
                body_parts.extend(paragraphs)
        
        return '\n'.join(body_parts)
    
    def _extract_section_title(self, section: Dict[str, Any]) -> str:
        """提取章节标题"""
        section_info = section.get('section', {})
        if not isinstance(section_info, dict):
            return ""
        
        # 处理index字段，可能是字符串、整数或其他类型
        index_raw = section_info.get('index', '')
        index = str(index_raw).strip() if index_raw is not None else ''
        
        # 处理name字段
        name_raw = section_info.get('name', '')
        name = str(name_raw).strip() if name_raw is not None else ''
        
        # 特殊处理：如果index是-1，通常表示没有编号
        if index == '-1':
            index = ''
        
        if index and name:
            return f"{index} {name}"
        elif name:
            return name
        elif index:
            return index
        
        return ""
    
    def _extract_paragraphs(self, section: Dict[str, Any]) -> List[str]:
        """提取段落"""
        paragraphs = section.get('p', [])
        if not isinstance(paragraphs, list):
            return []
        
        paragraph_texts = []
        for paragraph in paragraphs:
            if isinstance(paragraph, dict):
                text = paragraph.get('text', '').strip()
                if text:
                    paragraph_texts.append(text)
            elif isinstance(paragraph, str):
                text = paragraph.strip()
                if text:
                    paragraph_texts.append(text)
        
        return paragraph_texts
    
    def _extract_references(self, paper_json: Dict[str, Any]) -> str:
        """提取参考文献"""
        references = paper_json.get('reference', [])
        if not isinstance(references, list):
            return ""
        
        ref_parts = []
        for i, ref in enumerate(references):
            if isinstance(ref, dict):
                # 提取引用信息
                title = ref.get('title', '').strip()
                authors = ref.get('authors', [])
                year = ref.get('year', '').strip()
                
                ref_text = f"[{i+1}]"
                if title:
                    ref_text += f" {title}"
                if authors:
                    if isinstance(authors, list):
                        author_names = [str(author).strip() for author in authors if str(author).strip()]
                        if author_names:
                            ref_text += f", {', '.join(author_names)}"
                    else:
                        ref_text += f", {str(authors).strip()}"
                if year:
                    ref_text += f", {year}"
                
                ref_parts.append(ref_text)
            elif isinstance(ref, str):
                ref_parts.append(f"[{i+1}] {ref.strip()}")
        
        return '\n'.join(ref_parts)
