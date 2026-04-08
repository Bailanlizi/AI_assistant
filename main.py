from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_qwq import ChatQwen

import os
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

# ======================
# 【关键修复】导入 CORS 中间件
# ======================
from fastapi.middleware.cors import CORSMiddleware

# ======================
# 加载环境 & 模型
# ======================
load_dotenv()

llm = ChatQwen(
    model="qwen-turbo",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# ======================
# LCEL 链
# ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个个人AI助手，有记忆功能，回答简洁友好。"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

agent_chain = prompt | llm | StrOutputParser()

# ======================
# 短期记忆
# ======================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    agent_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ======================
# FastAPI 后端
# ======================
app = FastAPI()

# ======================
# 【关键修复】配置 CORS
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有，上线再改成具体地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法（包括 OPTIONS）
    allow_headers=["*"],  # 允许所有请求头
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        resp = chain_with_memory.invoke(
            {"input": req.message},
            config={"configurable": {"session_id": req.session_id}}
        )
        return {"code": 0, "answer": resp}
    except Exception as e:
        return {"code": -1, "error": str(e)}