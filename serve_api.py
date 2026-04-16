# serve_api.py
"""竞赛评审用 FastAPI 服务，按需启动。"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from src.agent.graph import run_agent
from src.utils.formatter import format_answer
from src.utils.config import KAFU_API_TOKEN

app = FastAPI(title="多模态客服智能体 API")
security = HTTPBearer()


class ChatRequest(BaseModel):
    question: str
    image: str | None = None  # Base64 编码图片，可选


class ChatResponse(BaseModel):
    answer: str


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != KAFU_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: str = Depends(verify_token),
):
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, run_agent, request.question, request.image
    )
    final_answer = result.get("final_answer", "")
    used_images = result.get("used_images", [])
    return ChatResponse(answer=format_answer(final_answer, used_images))


@app.get("/health")
async def health():
    return {"status": "ok"}
