from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

API_ROOT = "https://api.github.com"
BOT_LABEL = "linprogx-bot-seen"
TRIAGE_LABEL = "automated-triage"
BOT_EMAIL = "github-actions[bot]@users.noreply.github.com"
BOT_NAME = "github-actions[bot]"
MARKER_PREFIX = "linprogx-bot:item:"


@dataclass(frozen=True)
class WorkItem:
    kind: str
    marker_id: str
    issue_number: int
    issue_title: str
    issue_url: str
    opener: str
    author: str
    body: str
    source_url: str

    @property
    def branch(self) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", self.marker_id).strip("-").lower()
        return f"bot/triage-{safe}"

    @property
    def path(self) -> Path:
        if self.kind == "issue":
            return Path("triage/issues") / f"issue-{self.issue_number}.md"
        return Path("triage/comments") / f"{self.marker_id}.md"


def main() -> int:
    token = _env("GITHUB_TOKEN")
    repository = _env("GITHUB_REPOSITORY")
    limit = int(os.getenv("ISSUE_POLL_LIMIT", "5"))
    owner, repo = repository.split("/", 1)
    context = GithubContext(token=token, owner=owner, repo=repo)

    ensure_label(context, BOT_LABEL, "59636e", "Issue/comment has been seen by linprogx bot")
    ensure_label(context, TRIAGE_LABEL, "0e8a16", "Automated triage PR opened")
    configure_git()

    base_branch = context.default_branch()
    processed = 0
    for item in discover_work(context):
        if processed >= limit:
            break
        pr_url = create_tracking_pr(context, item, base_branch)
        comment_on_issue(context, item, pr_url)
        add_labels(context, item.issue_number, [BOT_LABEL, TRIAGE_LABEL])
        processed += 1

    print(f"processed {processed} issue/comment item(s)")
    return 0


class GithubContext:
    def __init__(self, *, token: str, owner: str, repo: str) -> None:
        self.token = token
        self.owner = owner
        self.repo = repo

    def api(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            f"{API_ROOT}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urlopen(request, timeout=30) as response:
            raw = response.read()
        if not raw:
            return None
        return json.loads(raw)

    def repo_path(self, suffix: str) -> str:
        return f"/repos/{self.owner}/{self.repo}{suffix}"

    def default_branch(self) -> str:
        repo = self.api("GET", self.repo_path(""))
        return str(repo["default_branch"])


def discover_work(context: GithubContext) -> list[WorkItem]:
    issues = context.api(
        "GET", context.repo_path("/issues?state=open&per_page=50&sort=updated&direction=desc")
    )
    work: list[WorkItem] = []
    for issue in issues:
        if "pull_request" in issue:
            continue
        issue_number = int(issue["number"])
        comments = context.api(
            "GET", context.repo_path(f"/issues/{issue_number}/comments?per_page=100")
        )
        processed_markers = markers_from_comments(comments)
        issue_marker = f"issue-{issue_number}"
        if issue_marker not in processed_markers:
            work.append(
                WorkItem(
                    kind="issue",
                    marker_id=issue_marker,
                    issue_number=issue_number,
                    issue_title=str(issue["title"]),
                    issue_url=str(issue["html_url"]),
                    opener=str(issue["user"]["login"]),
                    author=str(issue["user"]["login"]),
                    body=str(issue.get("body") or ""),
                    source_url=str(issue["html_url"]),
                )
            )
        for comment in comments:
            if _is_bot_user(comment["user"]):
                continue
            marker = f"comment-{comment['id']}"
            if marker in processed_markers:
                continue
            work.append(
                WorkItem(
                    kind="comment",
                    marker_id=marker,
                    issue_number=issue_number,
                    issue_title=str(issue["title"]),
                    issue_url=str(issue["html_url"]),
                    opener=str(issue["user"]["login"]),
                    author=str(comment["user"]["login"]),
                    body=str(comment.get("body") or ""),
                    source_url=str(comment["html_url"]),
                )
            )
    return work


def markers_from_comments(comments: list[dict[str, Any]]) -> set[str]:
    markers: set[str] = set()
    pattern = re.compile(rf"{re.escape(MARKER_PREFIX)}([a-zA-Z0-9._-]+)")
    for comment in comments:
        body = str(comment.get("body") or "")
        markers.update(match.group(1) for match in pattern.finditer(body))
    return markers


def create_tracking_pr(context: GithubContext, item: WorkItem, base_branch: str) -> str:
    run("git", "fetch", "origin", base_branch)
    run("git", "checkout", "-B", item.branch, f"origin/{base_branch}")
    item.path.parent.mkdir(parents=True, exist_ok=True)
    item.path.write_text(render_triage_note(item))
    run("git", "add", str(item.path))
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        existing = find_open_pr(context, item.branch)
        return existing or item.issue_url
    run("git", "commit", "-m", f"Add triage note for #{item.issue_number}")
    run("git", "push", "origin", item.branch, "--force-with-lease")
    existing = find_open_pr(context, item.branch)
    if existing:
        return existing
    payload = {
        "title": f"Triage {item.kind} for #{item.issue_number}: {item.issue_title}",
        "head": item.branch,
        "base": base_branch,
        "body": render_pr_body(item),
    }
    try:
        pr = context.api("POST", context.repo_path("/pulls"), payload)
        return str(pr["html_url"])
    except HTTPError as exc:
        if exc.code == 422:
            existing = find_open_pr(context, item.branch)
            if existing:
                return existing
        raise


def render_triage_note(item: WorkItem) -> str:
    clipped = item.body.strip()
    if len(clipped) > 4000:
        clipped = clipped[:4000] + "\n\n[truncated]"
    return textwrap.dedent(
        f"""\
        # Automated Triage: {item.kind.title()} for #{item.issue_number}

        - Issue: #{item.issue_number}
        - Title: {item.issue_title}
        - Opener: @{item.opener}
        - Source author: @{item.author}
        - Source URL: {item.source_url}

        ## Captured Text

        {clipped or "_No body text supplied._"}

        ## Bot Action

        The issue poll bot detected this {item.kind}, tagged @{item.author}, and opened this PR as a durable triage artifact.
        """
    )


def render_pr_body(item: WorkItem) -> str:
    return textwrap.dedent(
        f"""\
        Automated triage for {item.kind} on #{item.issue_number}.

        Source: {item.source_url}
        Opener: @{item.opener}
        Author to tag: @{item.author}

        This PR records the incoming repository activity so a maintainer can review it, edit the note, or replace it with a code/documentation fix.
        """
    )


def comment_on_issue(context: GithubContext, item: WorkItem, pr_url: str) -> None:
    body = textwrap.dedent(
        f"""\
        @{item.author} I picked this up in the repository polling workflow.

        Tracking PR: {pr_url}

        <!-- {MARKER_PREFIX}{item.marker_id} -->
        """
    )
    context.api("POST", context.repo_path(f"/issues/{item.issue_number}/comments"), {"body": body})


def find_open_pr(context: GithubContext, branch: str) -> str | None:
    head = f"{context.owner}:{branch}"
    pulls = context.api("GET", context.repo_path(f"/pulls?state=open&head={head}"))
    if pulls:
        return str(pulls[0]["html_url"])
    return None


def ensure_label(context: GithubContext, name: str, color: str, description: str) -> None:
    payload = {"name": name, "color": color, "description": description}
    try:
        context.api("POST", context.repo_path("/labels"), payload)
    except HTTPError as exc:
        if exc.code != 422:
            raise


def add_labels(context: GithubContext, issue_number: int, labels: list[str]) -> None:
    context.api("POST", context.repo_path(f"/issues/{issue_number}/labels"), {"labels": labels})


def configure_git() -> None:
    run("git", "config", "user.name", BOT_NAME)
    run("git", "config", "user.email", BOT_EMAIL)


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def _is_bot_user(user: dict[str, Any]) -> bool:
    return str(user.get("type")) == "Bot" or str(user.get("login", "")).endswith("[bot]")


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        msg = f"{name} is required"
        raise RuntimeError(msg)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
