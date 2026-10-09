"""Read-only GitHub REST client restricted to an allowlist of repositories."""

from __future__ import annotations

from typing import Any

import httpx

from triagebot.github.models import Comment, Issue, Label

API_VERSION = "2022-11-28"


class GitHubError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RepoNotAllowed(GitHubError):
    """The repository is not in the allowlist."""


class GitHubClient:
    def __init__(
        self,
        token: str,
        allowed_repos: tuple[str, ...],
        base_url: str = "https://api.github.com",
        timeout_s: float = 30,
        http: httpx.Client | None = None,
    ) -> None:
        self.allowed_repos = allowed_repos
        self.base_url = base_url
        self.http = http or httpx.Client(timeout=timeout_s)
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
        }

    def list_open_issues(self, repo: str, limit: int = 30) -> list[Issue]:
        """Open issues, newest first, excluding pull requests."""
        return self.list_issues(repo, state="open", limit=limit)

    def list_issues(self, repo: str, state: str = "all", limit: int = 500) -> list[Issue]:
        """Issues in the given state ("open", "closed" or "all"), newest first, no pull requests."""
        items = self._paginate(f"/repos/{repo}/issues", {"state": state, "sort": "created"}, repo)
        issues = [Issue.from_api(d) for d in items]
        return [i for i in issues if not i.is_pull_request][:limit]

    def get_issue(self, repo: str, number: int, max_comments: int = 10) -> Issue:
        issue = Issue.from_api(self._get(f"/repos/{repo}/issues/{number}", repo=repo))
        if issue.comment_count:
            raw = self._get(
                f"/repos/{repo}/issues/{number}/comments", {"per_page": max_comments}, repo
            )
            issue.comments = [Comment.from_api(c) for c in raw]
        return issue

    def list_labels(self, repo: str) -> list[Label]:
        return [Label.from_api(d) for d in self._paginate(f"/repos/{repo}/labels", {}, repo)]

    def _check_repo(self, repo: str) -> None:
        if repo not in self.allowed_repos:
            raise RepoNotAllowed(f"Repository {repo!r} is not in the allowlist")

    def _get(self, path: str, params: dict[str, Any] | None = None, repo: str = "") -> Any:
        self._check_repo(repo)
        return self._request(f"{self.base_url}{path}", params).json()

    def _paginate(
        self, path: str, params: dict[str, Any], repo: str, max_pages: int = 5
    ) -> list[dict[str, Any]]:
        """Fetch up to max_pages pages by following the Link header."""
        self._check_repo(repo)
        url: str | None = f"{self.base_url}{path}"
        query: dict[str, Any] | None = {**params, "per_page": 100}
        results: list[dict[str, Any]] = []
        for _ in range(max_pages):
            if url is None:
                break
            resp = self._request(url, query)
            results.extend(resp.json())
            url = resp.links.get("next", {}).get("url")
            query = None  # next URL already has the query
        return results

    def _request(self, url: str, params: dict[str, Any] | None) -> httpx.Response:
        try:
            resp = self.http.get(url, params=params, headers=self.headers)
        except httpx.HTTPError as exc:
            raise GitHubError(f"Request to GitHub failed: {exc}") from exc
        if resp.status_code >= 400:
            try:
                message = resp.json().get("message", resp.text)
            except ValueError:
                message = resp.text
            raise GitHubError(f"GitHub HTTP {resp.status_code}: {message}", resp.status_code)
        return resp
