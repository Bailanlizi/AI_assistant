AI 助手项目总结

项目概述

这是一个基于LangChain框架构建的、具备短期记忆功能的个人AI助手后端服务。项目通过FastAPI提供HTTP接口，使用阿里云Qwen模型作为底层大语言模型，支持多会话记忆管理。

技术架构

核心组件

• 模型层: LangChain ChatQwen (qwen3-max模型)

• 框架层: LangChain + FastAPI

• 记忆管理: LangChain RunnableWithMessageHistory

• 接口规范: RESTful API (JSON)

主要特性

1. 会话记忆功能: 支持基于session_id的多用户对话历史管理
2. 简洁友好: AI助手被设计为简洁友好的回答风格
3. CORS支持: 完整的前后端分离支持
4. 错误处理: 完整的异常捕获和错误响应机制

项目结构

```
app.py
├── 环境配置加载
├── 模型初始化 (ChatQwen)
├── LangChain管道构建
├── 记忆系统配置
├── FastAPI应用
│   ├── CORS中间件
│   ├── API路由
│   └── 请求/响应模型
```

核心功能

1. AI对话接口

• 端点: POST /chat

• 请求格式:
  {
    "message": "用户消息",
    "session_id": "用户会话ID(默认为default)"
  }
  
• 响应格式:
  {
    "code": 0,  // 0:成功, -1:错误
    "answer": "AI回复内容"
  }
  

2. 记忆系统

• 使用ChatMessageHistory存储对话历史

• 基于session_id隔离不同用户的对话记忆

• 支持短期记忆存储（内存存储）

环境要求

依赖包

fastapi>=0.104.0
langchain-core>=0.1.0
langchain-community>=0.0.0
langchain-qwq
python-dotenv>=1.0.0
uvicorn>=0.24.0


环境变量

DASHSCOPE_BASE_URL=你的DashScope API地址
DASHSCOPE_API_KEY=你的DashScope API密钥


快速开始

1. 安装依赖

pip install -r requirements.txt


2. 配置环境

创建.env文件，填入你的API配置

3. 启动服务

uvicorn app:app --reload --host 0.0.0.0 --port 8000


API使用示例

对话请求

curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，我是小明",
    "session_id": "user_123"
  }'


响应示例

{
  "code": 0,
  "answer": "你好小明！很高兴认识你。有什么我可以帮助你的吗？"
}


配置说明

CORS配置

• 开发环境：允许所有来源（allow_origins=["*"]）

• 生产环境：建议指定具体的前端地址

模型配置

• 模型：qwen3-max

• 可通过修改model参数切换不同版本的Qwen模型

扩展建议

1. 记忆存储优化

• 当前使用内存存储，可扩展为Redis或数据库存储

• 添加记忆过期机制

2. 功能增强

• 添加流式响应支持

• 集成工具调用（tools）

• 支持文件上传和处理

3. 生产部署

• 添加API认证

• 实现限流和频率控制

• 添加日志记录和监控

• 使用反向代理（Nginx）

4. 前端集成

// 前端调用示例
async function chatWithAI(message, sessionId = "default") {
  const response = await fetch("http://your-server/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId })
  });
  return response.json();
}


注意事项

1. 开发环境：当前CORS配置允许所有来源，生产环境需调整
2. API密钥：妥善保管DashScope API密钥，避免泄露
3. 会话管理：会话ID建议使用唯一标识符（如UUID）
4. 错误处理：客户端应检查code字段判断响应状态

项目优势

• 模块化设计：清晰的代码结构，易于维护和扩展

• 多会话支持：完善的会话隔离机制

• 开发友好：完善的CORS支持和错误处理

• 易于部署：基于FastAPI，部署简单

许可证

项目基于MIT许可证开源，可自由使用和修改。

这个项目为一个功能完善的AI助手后端服务，具有清晰的架构和良好的扩展性。通过简单的配置即可部署使用，适合作为个人助手、客服系统等场景的基础框架。
