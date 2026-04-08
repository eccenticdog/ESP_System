from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from graph.builder import get_chat_graph
from graph.memory import memory_service
from graph.streaming import reset_event_callback, set_event_callback


def _extract_final_answer(result: dict[str, Any]) -> str:
    for message in reversed(list(result.get("messages", []))):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return str(result.get("final_answer", "") or "")


class GraphChatService:
    async def run(
        self,
        *,
        user_id: str,
        session_id: str,
        user_query: str,
        emit_event,
    ) -> str:
        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
                "session_id": session_id,
                "emit_event": emit_event,
            }
        }
        token = set_event_callback(emit_event)
        try:
            graph = await get_chat_graph()
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "user_id": user_id,
                    "session_id": session_id,
                    "user_query": user_query,
                },
                config=config,
            )
            final_answer = _extract_final_answer(result)
            memory_service.schedule_memory_write(
                user_id=user_id,
                session_id=session_id,
                messages=list(result.get("messages", [])),
            )
        finally:
            reset_event_callback(token)
        return final_answer


graph_chat_service = GraphChatService()
