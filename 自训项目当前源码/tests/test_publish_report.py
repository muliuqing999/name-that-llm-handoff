import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("publish_report", Path(__file__).resolve().parents[1] / "scripts/publish_report.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class FakeAPI:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class PublishTests(unittest.TestCase):
    report = {"schema_version": 1, "run_id": "123-1", "status": "completed", "commit_sha": "abc", "metrics": {"accuracy": 0.7}}
    env = {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "abc", "GITHUB_REPOSITORY": "owner/repo"}

    def test_bootstrap_atomically_creates_results_only(self):
        api = FakeAPI([publisher.GitHubError(404, "GET", "ref"), {"sha": "tree"}, {"sha": "commit"}, {}])
        self.assertEqual(publisher.publish(api, self.report), "commit")
        tree = api.calls[1][2]
        self.assertEqual([x["path"] for x in tree["tree"]], ["reports/latest.json", "reports/runs/123-1.json"])
        self.assertNotIn("base_tree", tree)
        self.assertEqual(api.calls[2][2]["parents"], [])
        self.assertEqual(api.calls[-1], ("POST", "/git/refs", {"ref": "refs/heads/results", "sha": "commit"}))

    def test_update_preserves_previous_results_tree(self):
        api = FakeAPI([{"object": {"sha": "old"}}, {"tree": {"sha": "old-tree"}}, {"sha": "tree"}, {"sha": "commit"}, {}])
        publisher.publish(api, self.report)
        self.assertEqual(api.calls[2][2]["base_tree"], "old-tree")
        self.assertEqual(api.calls[3][2]["parents"], ["old"])
        self.assertEqual(api.calls[-1][2], {"sha": "commit", "force": False})

    def test_race_rebuilds_on_new_parent_without_force(self):
        api = FakeAPI([
            {"object": {"sha": "old"}}, {"tree": {"sha": "old-tree"}}, {"sha": "tree1"}, {"sha": "commit1"}, publisher.GitHubError(422, "PATCH", "ref"),
            {"object": {"sha": "new"}}, {"tree": {"sha": "new-tree"}}, {"sha": "tree2"}, {"sha": "commit2"}, {},
        ])
        self.assertEqual(publisher.publish(api, self.report, sleep=lambda _: None), "commit2")
        self.assertEqual(api.calls[7][2]["base_tree"], "new-tree")
        self.assertEqual(api.calls[8][2]["parents"], ["new"])

    def test_auth_error_is_not_swallowed_or_retried(self):
        api = FakeAPI([publisher.GitHubError(403, "GET", "ref")])
        with self.assertRaises(publisher.GitHubError):
            publisher.publish(api, self.report)
        self.assertEqual(len(api.calls), 1)

    def test_missing_and_stale_reports_become_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.json"
            missing = publisher.load_report(path, self.env)
            self.assertEqual(missing["status"], "failed")
            self.assertEqual(json.loads(path.read_text())["run_id"], "123-1")
            path.write_text(json.dumps(dict(self.report, run_id="122-1")), encoding="utf-8")
            self.assertEqual(publisher.load_report(path, self.env)["status"], "failed")

    def test_current_terminal_report_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.json"
            path.write_text(json.dumps(self.report), encoding="utf-8")
            self.assertEqual(publisher.load_report(path, self.env), self.report)

    def test_workflow_failure_overrides_completed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.json"
            path.write_text(json.dumps(self.report), encoding="utf-8")
            report = publisher.load_report(path, dict(self.env, TRAINING_OUTCOME="failure"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["training_step_outcome"], "failure")

    def test_rejects_bad_run_id_before_any_api_call(self):
        api = FakeAPI([])
        with self.assertRaises(ValueError):
            publisher.publish(api, dict(self.report, run_id="../main"))
        self.assertEqual(api.calls, [])


if __name__ == "__main__":
    unittest.main()
