import json
import logging
import os
from typing import Any, Dict

import grpc

from src.services.chat import generate_chat_exchange, generate_conversation_title

logger = logging.getLogger(__name__)

SERVICE_NAME = "neurolab.chat.v1.ChatService"


def _loads(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _dumps(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


async def _send_message(request: Dict[str, Any], _context) -> Dict[str, Any]:
    return await generate_chat_exchange(
        message=str(request.get("message", "")).strip(),
        subject_id=request.get("subject_id"),
        history=request.get("history"),
        current_title=request.get("current_title"),
        include_health_data=bool(request.get("include_health_data", True)),
    )


async def _generate_conversation_title(request: Dict[str, Any], _context) -> Dict[str, Any]:
    title = await generate_conversation_title(
        request.get("history") or [],
        subject_id=request.get("subject_id"),
        current_title=request.get("current_title"),
    )
    return {"title": title}


async def create_chat_grpc_server(bind_address: str | None = None):
    bind = bind_address or os.getenv("AI_CHAT_GRPC_BIND", "0.0.0.0:50051")
    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SERVICE_NAME,
                {
                    "SendMessage": grpc.unary_unary_rpc_method_handler(
                        _send_message,
                        request_deserializer=_loads,
                        response_serializer=_dumps,
                    ),
                    "GenerateConversationTitle": grpc.unary_unary_rpc_method_handler(
                        _generate_conversation_title,
                        request_deserializer=_loads,
                        response_serializer=_dumps,
                    ),
                },
            ),
        )
    )
    server.add_insecure_port(bind)
    logger.info(f"AI chat gRPC server bound to {bind}")
    return server
