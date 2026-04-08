import asyncio
import traceback
from collections.abc import AsyncGenerator

from infrastructure.logging.logger import logger
from schemas.request import ChatMessageRequest
from schemas.response import ContentKind
from services.graph_chat_service import graph_chat_service
from services.session_service import session_service
from utils.response_util import ResponseFactory
from utils.text_util import format_agent_update_html, format_tool_call_html


class MultiAgentService:
    """LangGraph based multi-agent chat service."""

    @classmethod
    async def process_task(cls, request: ChatMessageRequest, flag: bool) -> AsyncGenerator:
        try:
            user_id = request.context.user_id
            session_id = request.context.session_id or session_service.DEFAULT_SESSION_ID
            user_query = request.query
            event_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

            async def emit_event(kind: str, payload: str) -> None:
                await event_queue.put((kind, payload))

            graph_task = asyncio.create_task(
                graph_chat_service.run(
                    user_id=user_id,
                    session_id=session_id,
                    user_query=user_query,
                    emit_event=emit_event,
                )
            )

            while not graph_task.done() or not event_queue.empty():
                try:
                    event_kind, payload = await asyncio.wait_for(event_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                if event_kind == "tool":
                    text = format_tool_call_html(payload)
                elif event_kind == "agent":
                    text = format_agent_update_html(payload)
                else:
                    text = payload

                yield "data: " + ResponseFactory.build_text(
                    text,
                    ContentKind.PROCESS,
                ).model_dump_json() + "\n\n"

            agent_result = await graph_task
            if not agent_result:
                agent_result = "暂时没有生成可用回答。"

            for chunk in cls._chunk_text(agent_result):
                yield "data: " + ResponseFactory.build_text(
                    chunk,
                    ContentKind.ANSWER,
                ).model_dump_json() + "\n\n"

            archive_history = session_service.load_history(user_id, session_id)
            archive_history.append({"role": "user", "content": user_query})
            archive_history.append({"role": "assistant", "content": agent_result})
            session_service.save_history(user_id, session_id, archive_history)

            yield "data: " + ResponseFactory.build_finish().model_dump_json() + "\n\n"
        except Exception as e:
            logger.error(f"AgentService.process_query执行出错: {str(e)}")
            logger.debug(f"异常详情: {traceback.format_exc()}")

            yield "data: " + ResponseFactory.build_text(
                f"系统错误: {str(e)}",
                ContentKind.PROCESS,
            ).model_dump_json() + "\n\n"

            if flag:
                yield "data: " + ResponseFactory.build_text(
                    "正在尝试自动重试...",
                    ContentKind.PROCESS,
                ).model_dump_json() + "\n\n"

                async for item in MultiAgentService.process_task(request, flag=False):
                    yield item
            else:
                yield "data: " + ResponseFactory.build_finish().model_dump_json() + "\n\n"

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 96) -> list[str]:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
