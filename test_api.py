#!/usr/bin/env python3
"""测试 xlabapi.com API 的调用方式"""
import base64
from pathlib import Path
from anthropic import Anthropic

API_KEY = "sk-7753f81669019d125bc378d24b0480904087d350e7623c40ebdc6b057f384371"
BASE_URL = "https://xlabapi.com/v1"
MODEL = "claude-3-5-sonnet"

client = Anthropic(api_key=API_KEY, base_url=BASE_URL)

print("=== 测试 1: 纯文本消息 ===")
try:
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[
            {"role": "user", "content": "你好，请简单介绍一下你自己"}
        ]
    )
    print("✅ API 调用成功!")
    print("响应类型:", type(response))
    print("响应内容:", response)
    print("content 类型:", type(response.content))
    print("content 值:", response.content)
    if response.content:
        print("content[0] 类型:", type(response.content[0]))
        print("content[0] 值:", response.content[0])
        if hasattr(response.content[0], 'text'):
            print("text:", response.content[0].text[:200])
        else:
            print("没有 text 属性，所有属性:", dir(response.content[0]))
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试 2: 带 system 的消息 ===")
try:
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        system="你是一个专业的客服助手",
        messages=[
            {"role": "user", "content": "冰箱不制冷怎么办？"}
        ]
    )
    print("✅ 成功!")
    print("响应:", response.content[0].text[:200])
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n=== 测试 3: JSON 输出 ===")
try:
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system="返回JSON格式，不要加markdown代码块",
        messages=[
            {"role": "user", "content": '分析这个问题并返回JSON: {"question_type": "manual或policy", "product": "产品名或null"}\n\n问题：冰箱不制冷怎么办？'}
        ]
    )
    print("✅ 成功!")
    print("响应:", response.content[0].text[:200])
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n=== 测试 4: 多模态（文本+图片）===")
try:
    # 读取一张测试图片
    img_path = Path("手册/插图/冰箱_01.png")
    if img_path.exists():
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图片显示的是什么？"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}
                    ]
                }
            ]
        )
        print("✅ 成功!")
        print("响应:", response.content[0].text[:200])
    else:
        print("⏭️  跳过（图片不存在）")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n=== 测试 5: 多模态（system + 文本 + 图片）===")
try:
    img_path = Path("手册/插图/冰箱_01.png")
    if img_path.exists():
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system="你是产品手册助手",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "根据这张图片，描述产品特征"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}
                    ]
                }
            ]
        )
        print("✅ 成功!")
        print("响应:", response.content[0].text[:200])
    else:
        print("⏭️  跳过（图片不存在）")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n=== 测试完成 ===")
