import json
import logging
import os
from typing import Any, Dict, Optional

import grpc

logger = logging.getLogger(__name__)

SERVICE_NAME = "neurolab.context.v1.UserContextService"
DEFAULT_TARGET = os.getenv("BACKEND_USER_CONTEXT_GRPC_TARGET", "127.0.0.1:50052")


def _loads(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _dumps(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


async def fetch_user_context(
    *,
    auth_token: str,
    subject_id: Optional[str] = None,
    analysis_limit: int = 8,
) -> Dict[str, Any]:
    if not auth_token:
        raise RuntimeError("Authorization token is required for backend user context")

    call_target = os.getenv("BACKEND_USER_CONTEXT_GRPC_TARGET", DEFAULT_TARGET)
    async with grpc.aio.insecure_channel(call_target) as channel:
        stub = channel.unary_unary(
            f"/{SERVICE_NAME}/GetUserContext",
            request_serializer=_dumps,
            response_deserializer=_loads,
        )
        try:
            return await stub(
                {
                    "subject_id": subject_id,
                    "analysis_limit": max(1, min(int(analysis_limit), 20)),
                },
                metadata=(("authorization", auth_token),),
            )
        except grpc.aio.AioRpcError as exc:
            logger.error(
                "Failed to fetch backend user context via gRPC: code=%s details=%s",
                exc.code(),
                exc.details(),
            )
            raise RuntimeError(
                f"Backend user context retrieval failed: {exc.code().name} {exc.details()}"
            ) from exc
