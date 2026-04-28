import asyncio
import logging
import os

from src.config.database import ENABLE_DATABASES
from src.queue import get_redis
from src.services.database import db_service

logger = logging.getLogger(__name__)


def _is_required(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


async def _check_redis() -> None:
    if not _is_required("STARTUP_REQUIRE_REDIS", "true"):
        logger.info("Startup dependency check: Redis skipped")
        return

    redis = get_redis()

    try:
        await asyncio.to_thread(redis.ping)
        logger.info("Startup dependency check: Redis available")
    finally:
        try:
            await asyncio.to_thread(redis.close)
        except Exception:
            logger.debug("Redis client close failed during startup check")


async def _check_mongodb() -> None:
    if not ENABLE_DATABASES or not _is_required("STARTUP_REQUIRE_MONGODB", "true"):
        logger.info("Startup dependency check: MongoDB skipped")
        return

    if db_service.mongo_client is None:
        raise RuntimeError("MongoDB client is not initialized")

    await db_service.mongo_client.admin.command("ping")
    logger.info("Startup dependency check: MongoDB available")


async def _check_influxdb() -> None:
    if not ENABLE_DATABASES or not _is_required("STARTUP_REQUIRE_INFLUXDB", "true"):
        logger.info("Startup dependency check: InfluxDB skipped")
        return

    if db_service.influx_client is None:
        raise RuntimeError("InfluxDB client is not initialized")

    ok = await asyncio.to_thread(db_service.influx_client.ping)
    if not ok:
        raise RuntimeError("InfluxDB ping failed")

    logger.info("Startup dependency check: InfluxDB available")


async def validate_startup_dependencies() -> None:
    failures = []

    for check_name, check in (
        ("Redis", _check_redis),
        ("MongoDB", _check_mongodb),
        ("InfluxDB", _check_influxdb),
    ):
        try:
            await check()
        except Exception as exc:
            logger.error("Startup dependency check failed for %s: %s", check_name, exc)
            failures.append(f"{check_name}: {exc}")

    if failures:
        raise RuntimeError(
            "AI service startup blocked because required dependencies are unavailable: "
            + "; ".join(failures)
        )
