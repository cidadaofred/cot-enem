"""Console and JSON-lines file logging without secret values."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    *,
    level: str = "INFO",
    log_file: str | Path | None = None,
    structured_file: bool = True,
) -> logging.Logger:
    logger = logging.getLogger("cot_enem")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(console)
    if log_file:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter() if structured_file else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        logger.addHandler(file_handler)
    return logger
