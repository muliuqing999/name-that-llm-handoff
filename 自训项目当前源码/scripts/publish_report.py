"""Publish a small terminal report atomically; checkpoints stay on the GPU host."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MAX_REPORT_BYTES = 256 * 1024
TERMINAL_STATUSES = {"completed", "failed", "skipped"}


class GitHubError(RuntimeError):
    def __init__(self, status, method, path):
        self.status = status
        super().__init__(f"GitHub returned HTTP {status} for {method} {path}")


class GitHub:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("GITHUB_REPOSITORY must be owner/repository")
        if not token:
            raise ValueError("GITHUB_TOKEN is required for result publication")
        self.base = "https://api.github.com/repos/" + repository
        self.token = token

    def request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(self.base + path, data=data, method=method, headers={
            "Authorization": "Bearer " + self.token,
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "web-pro-gpu-runner",
        })
        try:
            with urlopen(req, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            # Never print response bodies or request headers containing credentials.
            raise GitHubError(exc.code, method, path) from None


def run_key(env):
    key = env.get("GITHUB_RUN_ID", "") + "-" + env.get("GITHUB_RUN_ATTEMPT", "")
    if not re.fullmatch(r"[0-9]+-[0-9]+", key):
        raise ValueError("GITHUB_RUN_ID and GITHUB_RUN_ATTEMPT must be numeric")
    return key


def load_report(path, env):
    """Reject a stale report and synthesize a failure if the trainer stopped early."""
    key = run_key(env)
    reason = None
    report = None
    if not path.is_file():
        reason = "The training step did not produce a report."
    elif path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Report exceeds 256 KiB; large outputs must remain local")
    else:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            reason = "The training report is not valid UTF-8 JSON."
        if reason is None and (not isinstance(report, dict)
                or report.get("run_id") != key
                or report.get("commit_sha") != env.get("GITHUB_SHA")
                or report.get("status") not in TERMINAL_STATUSES):
            reason = "The training report is stale or did not reach a terminal status."
        if reason is None and env.get("TRAINING_OUTCOME") in {"failure", "cancelled"} and report["status"] != "failed":
            reason = "The training step failed or was cancelled after writing its report."
    if reason:
        report = {
            "schema_version": 1,
            "status": "failed",
            "run_id": key,
            "commit_sha": env.get("GITHUB_SHA"),
            "run_url": f"https://github.com/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "training_step_outcome": env.get("TRAINING_OUTCOME", "unknown"),
            "error": reason,
            "metrics": {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return report


def publish(api, report, attempts=3, sleep=time.sleep):
    """One commit updates latest + immutable attempt path; never force a branch."""
    key = report["run_id"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", key):
        raise ValueError("Invalid report run_id")
    if report.get("status") not in TERMINAL_STATUSES:
        raise ValueError("Only terminal reports can be published")
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(content.encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Report exceeds 256 KiB")
    entries = [{"path": path, "mode": "100644", "type": "blob", "content": content}
               for path in ("reports/latest.json", f"reports/runs/{key}.json")]
    for attempt in range(attempts):
        head = None
        try:
            head = api.request("GET", "/git/ref/heads/results")["object"]["sha"]
        except GitHubError as exc:
            if exc.status != 404:
                raise
        tree_request = {"tree": entries}
        if head:
            tree_request["base_tree"] = api.request("GET", "/git/commits/" + head)["tree"]["sha"]
        tree = api.request("POST", "/git/trees", tree_request)["sha"]
        commit = api.request("POST", "/git/commits", {
            "message": f"Record experiment {key}: {report['status']}",
            "tree": tree,
            "parents": [head] if head else [],
        })["sha"]
        try:
            if head:
                api.request("PATCH", "/git/refs/heads/results", {"sha": commit, "force": False})
            else:
                api.request("POST", "/git/refs", {"ref": "refs/heads/results", "sha": commit})
            return commit
        except GitHubError as exc:
            if exc.status not in (409, 422) or attempt == attempts - 1:
                raise
            # A competing writer advanced/created results: rebuild on its new head.
            sleep(0.5 * (attempt + 1))
    raise RuntimeError("Result publication did not finish")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/latest.json"))
    args = parser.parse_args()
    try:
        api = GitHub(os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_TOKEN", ""))
        report = load_report(args.report, os.environ)
        commit = publish(api, report)
        print(f"Published {report['status']} report to results at {commit}")
        return 1 if report["status"] == "failed" else 0
    except Exception as exc:
        # These bounded errors do not include the token or HTTP response body.
        print(f"Report publication failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
