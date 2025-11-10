from flask import Flask, request, jsonify
from flask_cors import CORS
from config.config import AppConfig
from services.text_processor_service import TextProcessorService
from services.vllm_service import VllmService
from services.automatic_review_service import AutomaticReviewService
from models.paper_models import PaperRequest
import logging
import time
import json
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    
    # JSON配置
    app.config['JSON_AS_ASCII'] = False  
    app.config['JSON_Sort_KEYS'] = False
    app.json.ensure_ascii = False
    
    CORS(app)
    
    # 初始化服务
    config = AppConfig()
    vllm_service = VllmService(config)
    automatic_review_service = AutomaticReviewService(config, vllm_service)
    
    @app.route('/api/papers/health', methods=['GET'])
    def health():
        """健康检查接口"""
        return jsonify({"message": "Paper Review Backend is running!"}), 200
    
    @app.route('/api/papers/automatic-review', methods=['POST'])
    def automatic_review():
        """自动评审接口"""
        try:
            # 获取请求数据
            data = request.get_json()
            paper_request = PaperRequest.from_dict(data)
            
            logger.info("收到Automatic_Review评审请求")
            
            start_time = time.time()
            
            # 文本处理器
            text_processor = TextProcessorService(include_authors=paper_request.include_authors)
            
            # 获取完整论文内容
            paper_content = text_processor.process_paper_json(paper_request.paper_json, auto_truncate=False)
            
            # 记录原始论文内容长度
            logger.info(f"论文内容长度: {len(paper_content):,} 字符")
            
            # 检查是否可能超出模型上下文窗口
            if text_processor.tokenizer:
                try:
                    tokens = text_processor.tokenizer.encode(paper_content)
                    paper_token_length = len(tokens)
                    logger.info(f"论文 token 数量: {paper_token_length:,}")
                    if paper_token_length > text_processor.MAX_TOKENS:
                        logger.warning(f"警告: 论文 token 数量 ({paper_token_length}) 超过了设定的最大限制 ({text_processor.MAX_TOKENS})，可能会被模型截断")
                except Exception as e:
                    logger.warning(f"计算 token 数量时出错: {str(e)}")
            
            # 生成评审
            logger.info("开始生成评审，正在调用模型...")
            logger.info(f"使用温度设置: {paper_request.temperature}")
            logger.info(f"最大生成token设置: {paper_request.max_tokens}")
            review_result = automatic_review_service.generate_review(
                paper_content=paper_content, 
                temperature=paper_request.temperature,
                max_tokens=paper_request.max_tokens
            )
            
            if "error" not in review_result:
                # 构建完整的评审结果结构
                full_review_result = {
                    "result": {
                        "content": review_result.get("content", ""),
                        "type": review_result.get("type", "automatic_review"),
                        "source": review_result.get("source", "Automatic_Review")
                    }
                }
                
                # 格式化
                reviews = automatic_review_service.format_automatic_review_to_frontend(full_review_result)
                
                return jsonify(reviews), 200
            else:
                return jsonify([{
                    "name": "Error",
                    "content": f"评审生成失败: {review_result.get('error', '未知错误')}"
                }]), 500        
            
        except Exception as e:
            logger.error(f"Automatic_Review评审失败: {str(e)}")
            return jsonify([{
                "name": "Error",
                "content": f"评审生成失败: {str(e)}"
            }]), 500
    

    @app.route('/api/papers/test-vllm', methods=['GET'])
    def test_vllm():
        """测试vLLM连接"""
        try:
            result = vllm_service.generate_text(
                prompt="This is a test document about machine learning research. Please briefly review this document.",
                temperature=0.0,
                max_tokens=100
            )
            return jsonify({"message": "vLLM连接正常", "result": result}), 200
        except Exception as e:
            return jsonify({"error": f"vLLM连接失败: {str(e)}"}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8036, debug=True)
