# API 测试示例

Base URL=http://localhost:8036/api/papers

## 1. 盲评接口

```bash
POST /blind-review
```

### 请求示例:

```bash
curl -X POST http://localhost:8036/api/papers/blind-review \
  -H "Content-Type: application/json" \
  -d '{
    "paper_json": {
      "title": "Sample Research Paper",
      "abstract": ["This is a sample abstract for testing purposes."],
      "body": [
        {
          "section": {"index": "1", "name": "Introduction"},
          "p": [{"text": "This is the introduction section."}]
        }
      ]
    },
    "temperature": 0.0,
    "max_tokens": 8192,
    "include_authors": false
  }'
```

**响应:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "reviews": [
    {
      "review_id": "review_a",
      "sections": [
        {"name": "Summary", "content": "..."},
        ...
      ]
    },
    {
      "review_id": "review_b",
      "sections": [
        {"name": "Summary", "content": "..."},
        ...
      ]
    }
  ],
  "processing_time": 12.34
}
```

## 2. 提交用户选择

```bash
POST /blind-review/submit-selection
```
### 请求示例:

```bash
# 替换 session_id 为上一步返回的实际 session_id
curl -X POST http://localhost:8036/api/papers/blind-review/submit-selection \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "selected_review_id": "review_a"
  }'
```

**预期响应:**

```json
{
  "message": "选择已记录",
  "selected_model": "automatic_review"
}
```

## 3. 获取统计信息

```bash
GET /blind-review/statistics
```

### 请求示例:

```bash
curl -X GET http://localhost:8036/api/papers/blind-review/statistics
```

**预期响应:**

```json
{
  "total_selections": 5,
  "statistics": {
    "automatic_review": {
      "count": 2,
      "percentage": 40.0
    },
    "deep_review": {
      "count": 3,
      "percentage": 60.0
    }
  }
}
```