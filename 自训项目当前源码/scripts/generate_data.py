"""Resumable response collection. Plan is free; generate explicitly makes paid API calls."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
          "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_prompts(path):
    rows = read_jsonl(path)
    ids, texts = set(), set()
    for row in rows:
        if set(row) != {"prompt_id", "prompt", "split"} or row["split"] not in {"train", "validation"}:
            raise ValueError("Each prompt needs exactly prompt_id, prompt, split=train|validation")
        if not isinstance(row["prompt_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", row["prompt_id"]):
            raise ValueError("Use a stable safe prompt_id")
        normalized = " ".join(row["prompt"].split()).casefold()
        if not normalized or row["prompt_id"] in ids or normalized in texts:
            raise ValueError("Empty/duplicate prompt or cross-split prompt leakage")
        ids.add(row["prompt_id"])
        texts.add(normalized)
    if not rows:
        raise ValueError("Prompt file is empty")
    return rows


def enabled_models(catalog, require_keys=False):
    if set(catalog) != set(LABELS):
        raise ValueError("Catalog must list exactly the 18 competition labels")
    enabled = {}
    for label, model in catalog.items():
        allowed_fields = {"enabled", "api_kind", "base_url", "api_model", "api_key_env", "verified_exact_version", "version_evidence", "expected_response_models", "parameters", "api_version", "workspace_id"}
        if not isinstance(model, dict) or set(model) - allowed_fields:
            raise ValueError("Unknown model configuration field; credentials must only be environment variables")
        if not model.get("enabled"):
            continue
        if model.get("verified_exact_version") is not True or not model.get("version_evidence"):
            raise ValueError("Verify exact provider model version before enabling: " + label)
        if model.get("api_kind") not in {"openai_chat", "anthropic", "gemini"}:
            raise ValueError("Unsupported API adapter")
        endpoint = urlparse(model.get("base_url", ""))
        if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("Use a credential-free HTTPS base_url")
        if not model.get("api_model") or not model.get("expected_response_models"):
            raise ValueError("Set requested model and independently verified response model IDs")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", model.get("api_key_env", "")):
            raise ValueError("Use an environment variable name for API keys")
        if require_keys and not os.environ.get(model["api_key_env"]):
            raise ValueError("Missing API key environment variable: " + model["api_key_env"])
        params = model.get("parameters", {})
        allowed = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "reasoning_effort", "seed"} if model["api_kind"] == "openai_chat" else {"temperature", "top_p", "top_k", "max_tokens"} if model["api_kind"] == "anthropic" else {"temperature", "topP", "topK", "maxOutputTokens", "thinkingConfig"}
        if not isinstance(params, dict) or set(params) - allowed:
            raise ValueError("Unsupported generation parameters")
        enabled[label] = model
    return enabled


def fetch(model, prompt):
    endpoint = model["base_url"].rstrip("/")
    kind, key = model["api_kind"], os.environ[model["api_key_env"]]
    headers = {"Content-Type": "application/json"}
    params = model.get("parameters", {})
    if kind == "openai_chat":
        endpoint += "/chat/completions"
        headers["Authorization"] = "Bearer " + key
        body = {**params, "model": model["api_model"], "messages": [{"role": "user", "content": prompt}]}
    elif kind == "anthropic":
        endpoint += "/messages"
        headers.update({"Authorization": "Bearer " + key, "anthropic-version": model.get("api_version", "2023-06-01")})
        if model.get("workspace_id"):
            headers["anthropic-workspace-id"] = model["workspace_id"]
        body = {"max_tokens": 2048, **params, "model": model["api_model"], "messages": [{"role": "user", "content": prompt}]}
    else:
        endpoint += "/models/" + quote(model["api_model"], safe="") + ":generateContent"
        headers["x-goog-api-key"] = key
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": params}
    # Do not retry automatically after a timeout: the provider may already have billed it.
    with urlopen(Request(endpoint, data=json.dumps(body).encode(), headers=headers), timeout=180) as response:
        raw = json.load(response)
    if kind == "openai_chat":
        choice = raw["choices"][0]
        text, version, finish = choice["message"]["content"], raw.get("model"), choice.get("finish_reason")
    elif kind == "anthropic":
        text = "".join(p["text"] for p in raw["content"] if p.get("type") == "text")
        version, finish = raw.get("model"), raw.get("stop_reason")
    else:
        candidate = raw["candidates"][0]
        text = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought", False))
        version, finish = raw.get("modelVersion"), candidate.get("finishReason")
    return raw, text, version, finish


def collect(catalog, prompts, output, limit, workers=4):
    models = enabled_models(catalog, require_keys=True)
    if not models:
        raise ValueError("No verified models enabled; no API calls made")
    if not 1 <= limit <= 100000:
        raise ValueError("max_requests must be 1..100000")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be 1..16")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".writer.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        raw_dir = output / "raw"
        raw_dir.mkdir(exist_ok=True)
        jobs = []
        pending_count = 0
        for prompt in prompts:
            for label, model in models.items():
                job_id = digest({"prompt": prompt, "label": label, "model_config": model})
                target = raw_dir / (job_id + ".json")
                if target.exists():
                    continue
                pending_count += 1
                if len(jobs) < limit:
                    jobs.append((prompt, label, model, job_id, target))
        def do_job(job):
            prompt, label, model, job_id, target = job
            row = {**prompt, "model": label, "job_id": job_id, "model_config": model, "created_at": datetime.now(timezone.utc).isoformat()}
            try:
                raw, text, version, finish = fetch(model, prompt["prompt"])
                row.update(raw_response=raw, text=text, response_model=version, finish_reason=finish)
                if version not in model["expected_response_models"]:
                    row["status"] = "quarantined_model_version"
                elif finish not in {"stop", "end_turn", "STOP"} or not isinstance(text, str) or not text.strip():
                    row["status"] = "quarantined_incomplete_response"
                else:
                    row["status"] = "ok"
            except Exception as error:
                # Never persist request headers, credentials, or provider error bodies.
                row.update(status="failed", error_type=type(error).__name__)
                if isinstance(error, HTTPError):
                    row["http_status"] = error.code
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(target)
            print(json.dumps({"prompt_id": prompt["prompt_id"], "label": label, "status": row["status"]}), flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(do_job, job) for job in jobs]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        print(json.dumps({"requests_this_run": len(jobs), "all_jobs_attempted": pending_count <= limit, "stopped_at_limit": pending_count > limit}))
    finally:
        lock.unlink(missing_ok=True)


def export(output):
    raw = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "raw").glob("*.json"))]
    accepted, seen_jobs, seen_text = [], set(), {}
    counts = Counter()
    prompt_splits, prompt_content = {}, {}
    for row in raw:
        counts[row["status"]] += 1
        if row["status"] != "ok":
            continue
        key = (row["prompt_id"], row["model"])
        if key in seen_jobs:
            raise ValueError("Multiple configurations for a prompt/model; curate into separate dataset versions")
        seen_jobs.add(key)
        pkey = " ".join(row["prompt"].split()).casefold()
        old_split = prompt_splits.setdefault(pkey, row["split"])
        if old_split != row["split"]:
            raise ValueError("Prompt leaked across train/validation")
        if prompt_content.setdefault(row["prompt_id"], pkey) != pkey:
            raise ValueError("Prompt ID reused for different content")
        text_hash = hashlib.sha256(row["text"].strip().encode()).hexdigest()
        if text_hash in seen_text:
            if seen_text[text_hash] != row["split"]:
                raise ValueError("Exact response leaked across train/validation")
            counts["duplicate_response"] += 1
            continue
        seen_text[text_hash] = row["split"]
        accepted.append(row)
    for split in ("train", "validation"):
        with (output / (split + ".csv")).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "prompt_id", "model", "text"])
            writer.writeheader()
            for row in accepted:
                if row["split"] == split:
                    writer.writerow({"id": row["job_id"], **{k: row[k] for k in ("prompt_id", "model", "text")}})
    group_counts = Counter((r["split"], r["prompt_id"]) for r in accepted)
    report = {"statuses": dict(counts), "exported": len(accepted), "class_counts": dict(Counter(r["model"] for r in accepted)),
              "split_class_counts": {split: dict(Counter(r["model"] for r in accepted if r["split"] == split)) for split in ("train", "validation")},
              "complete_18_model_groups": sum(n == 18 for n in group_counts.values()), "incomplete_groups": sum(n != 18 for n in group_counts.values()),
              "missing_labels": sorted(set(LABELS) - {r["model"] for r in accepted}),
              "notes": "Raw text retained. Review length, refusal, language and near-duplicate distributions before training; export is not a quality approval."}
    (output / "quality-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "generate", "export"])
    parser.add_argument("--models", type=Path, default=Path("generation/models.local.json"))
    parser.add_argument("--prompts", type=Path, default=Path("generation/prompts.example.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/pilot-v1"))
    parser.add_argument("--max-requests", type=int, default=18)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action == "export":
        export(args.output)
        return
    catalog = json.loads(args.models.read_text(encoding="utf-8-sig"))
    prompts = load_prompts(args.prompts)
    models = enabled_models(catalog)
    if args.action == "plan":
        print(json.dumps({"prompts": len(prompts), "split_counts": dict(Counter(p["split"] for p in prompts)), "enabled_models": list(models),
              "missing_models": sorted(set(LABELS) - set(models)), "planned_requests": len(prompts) * len(models), "paid_calls_made": 0}, indent=2))
    else:
        collect(catalog, prompts, args.output, args.max_requests, args.workers)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # CLI errors are configuration-only; provider bodies are never printed.
        print(type(error).__name__ + ": " + str(error) if isinstance(error, ValueError) else type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
