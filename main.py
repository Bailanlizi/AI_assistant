
# 核心
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langgraph.checkpoint.memory import InMemorySaver

# 导入LCEL核心组件
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

# LCEL的短期记忆存储
from langchain_community.chat_message_histories import ChatMessageHistory # 会话历史存储
from langchain_core.runnables.history import RunnableWithMessageHistory # 核心包装器，创建带有消息历史的可运行对象

# 语言模型——社区集成
from langchain_qwq import ChatQwen

# 构建agent
from langchain.agents import create_agent

import os # 导入os模块，用于访问环境变量
from dotenv import load_dotenv # 导入load_dotenv函数，用于加载环境变量

load_dotenv()  # 加载环境变量


# 创建一个ChatQwen实例，指定模型名称、API基础URL和API密钥
llm = ChatQwen(
    model = "qwen-turbo",
    base_url = os.getenv("DASHSCOPE_BASE_URL"),
    api_key = os.getenv("DASHSCOPE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个个人AI助手，有记忆功能，回答简洁友好，能调用工具解决简单问题。"),
    MessagesPlaceholder("history"), # 历史占位符 ← 关键！
    ("human", "{input}")

])

# 原始链
agent_chain = (
    RunnableSequence(prompt,llm,StrOutputParser()
    )
)
# agent_chain = prompt | llm | StrOutputParser()


# 带记忆的LCEL
# 创建会话管理器
store = {} # 存储会话历史的字典，键为session_id，值为ChatMessageHistory对象
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory() # 创建一个新的会话历史对象
    return store[session_id]
    
chain_with_memory = RunnableWithMessageHistory(
    agent_chain, # 核心链条：原始链
    get_session_history, # 会话管理器：根据session_id获取会话历史
    input_messages_key="input", #
    history_messages_key = "history" # 

)

# 多轮对话基础版
def chat_loop():
    """简单的多轮对话循环"""
    session_id = "user123"  # 固定会话ID
    print("🤖 AI助手已启动！输入 '退出' 结束对话")
    print("=" * 40)
    
    while True:
        # 获取用户输入
        user_input = input("\n👤 你: ").strip()
        
        if not user_input:
            print("⚠️ 请输入内容")
            continue
        
        # 检查退出命令
        if user_input.lower() in ['退出', 'exit', 'quit', 'bye']:
            print("👋 再见！")
            break
        
        try:
            # 调用AI
            print("🤔 AI思考中...", end="", flush=True)
            response = chain_with_memory.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
                timeout=30
            )
            
            # 显示AI回复
            print("\r" + " " * 20, end="\r")  # 清空"思考中"
            print(f"🤖 AI: {response}")
            
        except Exception as e:
            print(f"\r❌ 出错了: {e}")

# 运行
if __name__ == "__main__":
    chat_loop()
