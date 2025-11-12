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
import random
import uuid
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 存储盲评会话信息（内存存储，生产环境应使用数据库）
blind_review_sessions = {}
user_selections = []

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
    
    @app.route('/api/papers/blind-review', methods=['POST'])
    def blind_review():
        """盲评接口 - 同时调用两个模型并随机打乱顺序"""
        try:
            # 获取请求数据
            data = request.get_json()
            paper_request = PaperRequest.from_dict(data)
            
            logger.info("收到盲评请求")
            
            start_time = time.time()
            
            # 文本处理器
            text_processor = TextProcessorService(include_authors=paper_request.include_authors)
            
            # 获取完整论文内容
            paper_content = text_processor.process_paper_json(paper_request.paper_json, auto_truncate=False)
            
            # 记录原始论文内容长度
            logger.info(f"论文内容长度: {len(paper_content):,} 字符")
            
            # 并行生成两个评审
            logger.info("开始生成两个模型的评审...")
            
            # 1. 生成 Automatic_Review 评审
            logger.info("正在调用 Automatic_Review 模型...")
            automatic_review_result = automatic_review_service.generate_review(
                paper_content=paper_content, 
                temperature=paper_request.temperature,
                max_tokens=paper_request.max_tokens
            )
            
            # 2. 生成 Deep Review 评审
            logger.info("正在调用 Deep Review 模型...")
            deep_review_result = automatic_review_service.generate_deep_review(
                paper_content=paper_content,
                temperature=paper_request.temperature,
                max_tokens=paper_request.max_tokens
            )
            
            # 格式化评审结果
            if "error" not in automatic_review_result:
                full_automatic_review = {
                    "result": {
                        "content": automatic_review_result.get("content", ""),
                        "type": automatic_review_result.get("type", "automatic_review"),
                        "source": automatic_review_result.get("source", "Automatic_Review")
                    }
                }
                automatic_reviews = automatic_review_service.format_automatic_review_to_frontend(full_automatic_review)
            else:
                automatic_reviews = [{
                    "name": "Error",
                    "content": f"Automatic_Review 评审失败: {automatic_review_result.get('error', '未知错误')}"
                }]
            
            if "error" not in deep_review_result:
                full_deep_review = {
                    "result": {
                        "content": deep_review_result.get("content", ""),
                        "type": deep_review_result.get("type", "deep_review"),
                        "source": deep_review_result.get("source", "deep-review-7b")
                    }
                }
                deep_reviews = automatic_review_service.format_deep_review_to_frontend(full_deep_review)
            else:
                deep_reviews = [{
                    "name": "Error",
                    "content": f"Deep Review 评审失败: {deep_review_result.get('error', '未知错误')}"
                }]
            
            # 创建会话ID
            session_id = str(uuid.uuid4())
            
            # 随机打乱顺序
            reviews_list = [
                {
                    "model": "automatic_review",
                    "reviews": automatic_reviews
                },
                {
                    "model": "deep_review",
                    "reviews": deep_reviews
                }
            ]
            
            random.shuffle(reviews_list)
            
            # 存储会话信息（记录哪个是哪个模型）
            blind_review_sessions[session_id] = {
                "timestamp": datetime.now().isoformat(),
                "review_a": {
                    "model": reviews_list[0]["model"],
                    "position": "A"
                },
                "review_b": {
                    "model": reviews_list[1]["model"],
                    "position": "B"
                }
            }
            
            logger.info(f"盲评会话 {session_id} 创建成功")
            logger.info(f"Review A: {reviews_list[0]['model']}, Review B: {reviews_list[1]['model']}")
            
            # 构建返回结果
            result = {
                "session_id": session_id,
                "reviews": [
                    {
                        "review_id": "review_a",
                        "sections": reviews_list[0]["reviews"]
                    },
                    {
                        "review_id": "review_b",
                        "sections": reviews_list[1]["reviews"]
                    }
                ],
                "processing_time": time.time() - start_time
            }
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"盲评失败: {str(e)}")
            return jsonify({
                "error": f"盲评生成失败: {str(e)}"
            }), 500
    
    @app.route('/api/papers/blind-review/submit-selection', methods=['POST'])
    def submit_selection():
        """提交用户选择的评审"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            selected_review_id = data.get('selected_review_id')  # 'review_a' 或 'review_b'
            
            if not session_id or not selected_review_id:
                return jsonify({"error": "缺少必要参数"}), 400
            
            # 查找会话信息
            session = blind_review_sessions.get(session_id)
            if not session:
                return jsonify({"error": "会话不存在"}), 404
            
            # 确定用户选择的是哪个模型
            if selected_review_id == "review_a":
                selected_model = session["review_a"]["model"]
            elif selected_review_id == "review_b":
                selected_model = session["review_b"]["model"]
            else:
                return jsonify({"error": "无效的评审ID"}), 400
            
            # 记录选择
            selection_record = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "selected_review_id": selected_review_id,
                "selected_model": selected_model,
                "session_info": session
            }
            
            user_selections.append(selection_record)
            
            logger.info(f"用户选择记录成功: 会话 {session_id}, 选择 {selected_review_id} ({selected_model})")
            
            return jsonify({
                "message": "选择已记录",
                "selected_model": selected_model
            }), 200
            
        except Exception as e:
            logger.error(f"记录用户选择失败: {str(e)}")
            return jsonify({"error": f"记录失败: {str(e)}"}), 500
    
    @app.route('/api/papers/blind-review/statistics', methods=['GET'])
    def get_statistics():
        """获取盲评统计信息"""
        try:
            total_selections = len(user_selections)
            
            if total_selections == 0:
                return jsonify({
                    "total_selections": 0,
                    "statistics": {}
                }), 200
            
            # 统计每个模型被选择的次数
            model_counts = {}
            for selection in user_selections:
                model = selection["selected_model"]
                model_counts[model] = model_counts.get(model, 0) + 1
            
            # 计算百分比
            statistics = {}
            for model, count in model_counts.items():
                statistics[model] = {
                    "count": count,
                    "percentage": (count / total_selections) * 100
                }
            
            return jsonify({
                "total_selections": total_selections,
                "statistics": statistics
            }), 200
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return jsonify({"error": f"获取统计失败: {str(e)}"}), 500

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