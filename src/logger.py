"""JSONL logging for retrieval and answer traces, without raw question/answer text."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from logging.handlers import RotatingFileHandler

from src import config

LOGGER_NAME = "rag"

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("api_key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"(?<![\d-])[6-9]\d{4}[\s-]\d{5}(?![\d-])")),
    ("phone", re.compile(r"(?<![\d-])[6-9]\d{9}(?![\d-])")),
    ("account", re.compile(r"(?<!\d)\d{6,18}(?!\d)")),
)

REDACTED = "[redacted:{kind}:{digest}]"

SENSITIVE_TEXT_FIELDS = frozenset(
    {"question", "normalized_question", "answer", "raw_question", "raw_answer"}
)

_COMBINED_PATTERN = re.compile(
    "|".join(
        f"(?P<{kind}_{index}>{pattern.pattern})"
        for index, (kind, pattern) in enumerate(PII_PATTERNS)
    )
)

class _RedactingFormatter(logging.Formatter):
    """Redact PII and API-key shapes from every formatted log record."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_pii(super().format(record))


_configured = False


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Create logs/ and attach a stream handler plus a rotating file handler."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    if _configured:
        return logger
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = _RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        config.LOG_DIR / "pipeline.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    logger.propagate = False
    _configured = True
    return logger


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _replace_match(match: re.Match[str]) -> str:
    for name, value in match.groupdict().items():
        if value is not None:
            kind = name.rsplit("_", 1)[0]
            return REDACTED.format(kind=kind, digest=_digest(value))
    return match.group(0)


def redact_pii(text: str) -> str:
    """Replace PII-looking spans with a hashed marker so raw PII is never written."""
    if not text:
        return text
    return _COMBINED_PATTERN.sub(_replace_match, text)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _sanitize_record(value: object) -> object:
    if isinstance(value, dict):
        sanitized: dict = {}
        for key, item in value.items():
            if key in SENSITIVE_TEXT_FIELDS and isinstance(item, str):
                sanitized[f"{key}_sha256"] = _fingerprint(item)
                sanitized[f"{key}_chars"] = len(item)
            else:
                sanitized[key] = _sanitize_record(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_record(item) for item in value]
    return value


def _redact_value(value: object) -> object:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {
            key: item
            if isinstance(key, str) and key.endswith(("_sha256", "_chars"))
            else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def _append_jsonl(filename: str, record: dict) -> None:
    path = config.LOG_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_record(record)
    payload = json.dumps(_redact_value(sanitized), ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def log_retrieval(record: dict) -> None:
    """Append one redacted JSON line without the raw question or context."""
    _append_jsonl("retrieval.jsonl", record)


def log_answer(record: dict) -> None:
    """Append one redacted JSON line without the raw question or full answer."""
    _append_jsonl("answers.jsonl", record)
