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
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL 数据库配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '39.105.31.73'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'Cbers123123'),
    'database': os.getenv('DB_NAME', 'DH_Review')
}

def clean_section_content(content):
    if not content:
        return ""
    cleaned = content.strip()
    if cleaned.endswith('}') and cleaned.count('}') == 1 and '{' not in cleaned:
        cleaned = cleaned[:-1].rstrip()
    return cleaned

def filter_missing_sections(sections):
    filtered = []
    for section in sections:
        content = section.get('content')
        if '信息未找到' in (content or ''):
            continue
        sanitized = dict(section)
        sanitized['content'] = clean_section_content(content)
        filtered.append(sanitized)
    return filtered

def prepare_deep_review_sections(sections):
    desired_order = ["Summary", "Strengths", "Weaknesses", "Decision"]
    prepared = {}
    for section in sections:
        name = (section.get('name') or '').strip()
        if name in desired_order:
            content = section.get('content', '')
            if content:
                prepared[name] = content
    return [{"name": name, "content": prepared[name]} for name in desired_order if name in prepared]

@contextmanager
def get_db():
    """获取数据库连接"""
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """初始化数据库表"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blind_review_sessions (
                    session_id VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME,
                    review_a_model VARCHAR(255),
                    review_b_model VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_selections (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36),
                    timestamp DATETIME,
                    selected_review_id VARCHAR(50),
                    selected_model VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES blind_review_sessions(session_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_review_sessions (
                    session_id VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME,
                    review_a_model VARCHAR(255),
                    review_b_model VARCHAR(255),
                    review_a_raw_output TEXT,
                    review_b_raw_output TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            cursor.close()
            logger.info("数据库表初始化成功")
    except Error as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        raise

# 存储盲评会话信息（内存缓存，用于快速查询当前会话）
blind_review_sessions = {}

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
    
    # 初始化数据库
    init_db()
    
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
                reviews = filter_missing_sections(reviews)

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
                automatic_reviews = filter_missing_sections(automatic_reviews)

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
                deep_reviews = filter_missing_sections(deep_reviews)
                deep_reviews = prepare_deep_review_sections(deep_reviews)
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
            
            # 存储会话信息到 MySQL 数据库
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO blind_review_sessions 
                    (session_id, timestamp, review_a_model, review_b_model)
                    VALUES (%s, %s, %s, %s)
                ''', (session_id, datetime.now(), 
                      reviews_list[0]["model"], reviews_list[1]["model"]))
                conn.commit()
                cursor.close()
            
            # 同时保持内存缓存
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
            selected_review_id = data.get('selected_review_id')
            
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
            
            # 记录选择到 MySQL 数据库
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_selections 
                    (session_id, timestamp, selected_review_id, selected_model)
                    VALUES (%s, %s, %s, %s)
                ''', (session_id, datetime.now(), 
                      selected_review_id, selected_model))
                conn.commit()
                cursor.close()
            
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
            with get_db() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # 获取总选择数
                cursor.execute('SELECT COUNT(*) as count FROM user_selections')
                total_result = cursor.fetchone()
                total_selections = total_result['count']
                
                if total_selections == 0:
                    cursor.close()
                    return jsonify({
                        "total_selections": 0,
                        "statistics": {}
                    }), 200
                
                # 统计每个模型被选择的次数
                cursor.execute('''
                    SELECT selected_model, COUNT(*) as count 
                    FROM user_selections 
                    GROUP BY selected_model
                ''')
                model_results = cursor.fetchall()
                
                statistics = {}
                for row in model_results:
                    model = row['selected_model']
                    count = row['count']
                    statistics[model] = {
                        "count": count,
                        "percentage": (count / total_selections) * 100
                    }
                
                cursor.close()
                return jsonify({
                    "total_selections": total_selections,
                    "statistics": statistics
                }), 200
            
        except Error as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return jsonify({"error": f"获取统计失败: {str(e)}"}), 500

    @app.route('/api/papers/test-blind-review', methods=['POST'])
    def test_blind_review():
        """测试盲评接口 - 存储原始模型输出"""
        try:
            data = request.get_json()
            paper_request = PaperRequest.from_dict(data)
            
            logger.info("收到测试盲评请求")
            
            start_time = time.time()
            
            text_processor = TextProcessorService(include_authors=paper_request.include_authors)
            paper_content = text_processor.process_paper_json(paper_request.paper_json, auto_truncate=False)
            
            logger.info(f"论文内容长度: {len(paper_content):,} 字符")
            logger.info("开始生成两个模型的评审...")
            
            # 1. 生成 Automatic_Review 评审
            logger.info("正在调用 Automatic_Review 模型...")
            automatic_review_result = automatic_review_service.generate_review(
                paper_content=paper_content, 
                temperature=paper_request.temperature,
                max_tokens=paper_request.max_tokens
            )
            automatic_raw_output = automatic_review_result.get("content", "")
            
            # 2. 生成 Deep Review 评审
            logger.info("正在调用 Deep Review 模型...")
            deep_review_result = automatic_review_service.generate_deep_review(
                paper_content=paper_content,
                temperature=paper_request.temperature,
                max_tokens=paper_request.max_tokens
            )
            deep_raw_output = deep_review_result.get("content", "")
            
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
                automatic_reviews = filter_missing_sections(automatic_reviews)
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
                deep_reviews = filter_missing_sections(deep_reviews)
                deep_reviews = prepare_deep_review_sections(deep_reviews)
            else:
                deep_reviews = [{
                    "name": "Error",
                    "content": f"Deep Review 评审失败: {deep_review_result.get('error', '未知错误')}"
                }]
            
            session_id = str(uuid.uuid4())
            
            reviews_list = [
                {
                    "model": "automatic_review",
                    "reviews": automatic_reviews,
                    "raw_output": automatic_raw_output
                },
                {
                    "model": "deep_review",
                    "reviews": deep_reviews,
                    "raw_output": deep_raw_output
                }
            ]
            
            random.shuffle(reviews_list)
            
            # 存储会话信息和原始输出到数据库
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO test_review_sessions 
                    (session_id, timestamp, review_a_model, review_b_model, 
                     review_a_raw_output, review_b_raw_output)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (session_id, datetime.now(), 
                      reviews_list[0]["model"], reviews_list[1]["model"],
                      reviews_list[0]["raw_output"], reviews_list[1]["raw_output"]))
                conn.commit()
                cursor.close()
            
            logger.info(f"测试盲评会话 {session_id} 创建成功")
            logger.info(f"Review A: {reviews_list[0]['model']}, Review B: {reviews_list[1]['model']}")
            
            result = {
                "session_id": session_id,
                "reviews": [
                    {
                        "review_id": "review_a",
                        "sections": reviews_list[0]["reviews"],
                        "raw_output": reviews_list[0]["raw_output"]
                    },
                    {
                        "review_id": "review_b",
                        "sections": reviews_list[1]["reviews"],
                        "raw_output": reviews_list[1]["raw_output"]
                    }
                ],
                "processing_time": time.time() - start_time
            }
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"测试盲评失败: {str(e)}")
            return jsonify({
                "error": f"测试盲评生成失败: {str(e)}"
            }), 500
    
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