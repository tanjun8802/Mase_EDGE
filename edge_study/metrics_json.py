"""ADB / JSON metrics helpers."""


def _metrics_payload_ready(data: dict) -> bool:
    st = data.get("status")
    if st in ("done", "error"):
        return True
    return isinstance(data, dict) and "latency_p95_ms" in data and "top1_acc" in data


def _adb_cat_stdout_not_metrics_json(raw: bytes) -> bool:
    if not (raw or b"").strip():
        return True
    if raw.lstrip().startswith(b"cat:"):
        return True
    if b"No such file or directory" in raw:
        return True
    if b"Permission denied" in raw:
        return True
    return False
