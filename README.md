# Hammer Review Backend

学术论文自动评审后端服务，集成 Automatic_Review 项目，提供基于大语言模型的论文评审功能。

## 功能特性

- **自动评审生成**：基于 vLLM 和大语言模型自动生成论文评审
- **结构化输出**：按照 Summary、Strengths、Weaknesses、Decision 四个部分输出
- **JSON 解析**：支持复杂的论文 JSON 格式解析
- **匿名评审**：可选择是否包含作者信息，支持双盲评审
- **性能优化**：Token 级别的文本处理和智能截断
- **RESTful API**：提供标准的 HTTP API 接口

## 系统架构

```
Hammer_review_backend/
├── app.py                          # Flask 应用主入口
├── config/
│   └── config.py                   # 配置管理
├── models/
│   ├── paper_models.py            # 论文相关数据模型
│   └── vllm_models.py             # vLLM 相关数据模型
├── services/
│   ├── automatic_review_service.py # 自动评审服务
│   ├── text_processor_service.py   # 文本处理服务
│   └── vllm_service.py             # vLLM 调用服务
└── README.md
```

## 环境要求

- Python 3.8+
- Flask 2.0+
- vLLM 服务（需单独部署）
- transformers（可选，用于 token 级别处理）


```bash
# vLLM 服务配置
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL_NAME=your-model-name
VLLM_TIMEOUT=300

# 应用配置
FLASK_ENV=development
FLASK_DEBUG=True
```

### 4. 配置 Automatic_Review 项目

确保 `Automatic_Review` 项目位于父目录：

```
parent_directory/
├── Automatic_Review/
│   ├── generation/
│   │   └── prompts/
│   │       └── prompt_generate_review_v2.txt
│   └── evaluation/
└── Hammer_review_backend/
```

## 运行

### 开发模式

```bash
python app.py
```

服务将在 `http://0.0.0.0:8036` 启动。

### 生产模式

```bash
gunicorn -w 4 -b 0.0.0.0:8036 app:app
```

## API 文档

### 1. 健康检查

**接口**: `GET /api/papers/health`

**响应**:
```json
{
  "message": "Paper Review Backend is running!"
}
```

### 2. 自动评审

**接口**: `POST /api/papers/automatic-review`

**请求体**:
```json
{
  "paper_json": {
    "title": "论文标题",
    "author": [
      {"name": "作者1"},
      {"name": "作者2"}
    ],
    "abstract": ["摘要内容"],
    "body": [
      {
        "section": {"index": "1", "name": "Introduction"},
        "p": [{"text": "段落内容"}]
      }
    ],
    "reference": []
  },
  "include_authors": false,
  "temperature": 0.0,
  "max_tokens": 8192
}
```

**参数说明**:
- `paper_json`: 论文的 JSON 格式数据（必填）
- `include_authors`: 是否包含作者信息，默认 `false`（推荐双盲评审）
- `temperature`: 生成温度，范围 0.0-1.0，默认 0.0（确定性输出）
- `max_tokens`: 最大生成 token 数，默认 8192

**响应**:
```json
[
  {
    "name": "Summary",
    "content": "论文概述内容..."
  },
  {
    "name": "Strengths",
    "content": "论文优势分析..."
  },
  {
    "name": "Weaknesses",
    "content": "论文不足之处..."
  },
  {
    "name": "Decision",
    "content": "评审决策建议..."
  }
]
```

### 3. 测试 vLLM 连接

**接口**: `GET /api/papers/test-vllm`

**响应**:
```json
{
  "message": "vLLM连接正常",
  "result": "测试生成的内容"
}
```

## 配置说明

### VllmService 配置

在 `config/config.py` 中配置 vLLM 服务：

```python
class VllmConfig:
    base_url: str = "http://localhost:8000"
    model_name: str = "your-model-name"
    timeout: int = 300
```

### TextProcessorService 配置

```python
class TextProcessorService:
    MAX_LENGTH = 82768   # 最大字符长度
    MAX_TOKENS = 32000   # 最大 token 数量
```

## 核心服务说明

### 1. AutomaticReviewService

负责集成 Automatic_Review 项目功能：
- 加载评审 prompt 模板
- 调用 LLM 生成评审
- 格式化评审结果

### 2. TextProcessorService

处理论文文本：
- JSON 到文本的转换
- Token 级别的文本截断
- 支持匿名评审（排除作者信息）

### 3. VllmService

管理与 vLLM 服务的通信：
- 文本生成（同步/流式）
- 模型预热
- 错误处理

