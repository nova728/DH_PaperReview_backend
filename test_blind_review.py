import json
import requests
import sys

def test_blind_review():
    """测试盲评接口"""
    # 读取测试 JSON 文件
    with open('test_request_json.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # API 端点
    url = "http://localhost:8036/api/papers/blind-review"
    
    # 构建请求数据
    request_data = {
        "paper_json": test_data["paper_json"],
        "temperature": test_data.get("temperature", 0.0),
        "max_tokens": test_data.get("max_tokens", 8192),
        "include_authors": False  # 盲评模式，不包含作者信息
    }
    
    print("=" * 80)
    print("开始测试盲评接口...")
    print("=" * 80)
    print(f"论文标题: {test_data['paper_json']['title']}")
    print(f"作者数量: {len(test_data['paper_json']['author'])}")
    print(f"章节数量: {len(test_data['paper_json']['body'])}")
    print(f"温度: {request_data['temperature']}")
    print(f"最大Token: {request_data['max_tokens']}")
    print("=" * 80)
    
    try:
        # 发送请求
        print("\n正在发送请求到服务器...")
        response = requests.post(url, json=request_data, timeout=600)
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            
            print("\n✓ 请求成功!")
            print("=" * 80)
            print(f"会话ID: {result.get('session_id')}")
            print(f"处理时间: {result.get('processing_time', 0):.2f} 秒")
            print("=" * 80)
            
            # 打印两个评审的摘要
            reviews = result.get('reviews', [])
            for i, review in enumerate(reviews, 1):
                print(f"\n评审 {chr(64 + i)} ({review['review_id']}):")
                print("-" * 80)
                
                sections = review.get('sections', [])
                for section in sections[:3]:  # 只打印前3个部分
                    name = section.get('name', 'Unknown')
                    content = section.get('content', '')
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"\n{name}:")
                    print(content_preview)
                
                if len(sections) > 3:
                    print(f"\n... 还有 {len(sections) - 3} 个部分")
            
            # 保存完整结果到文件
            output_file = 'blind_review_result.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print("\n" + "=" * 80)
            print(f"✓ 完整结果已保存到: {output_file}")
            print("=" * 80)
            
        else:
            print(f"\n✗ 请求失败! 状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            sys.exit(1)
            
    except requests.exceptions.Timeout:
        print("\n✗ 请求超时! 论文可能太长或服务器响应慢")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n✗ 连接失败! 请确保服务器正在运行 (http://localhost:8036)")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_blind_review()
