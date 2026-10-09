"""Fake GitHub API backed by JSON fixtures."""

import json
from pathlib import Path

import httpx
import pytest

from triagebot.github.client import GitHubClient

FIXTURES = Path(__file__).parent / "fixtures"
REPO = "me/sandbox"
BASE = "https://api.github.test"


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


class FakeGitHub:
    """Serves fixture data by path and records requests."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        issue_3 = load("issues_page1.json")[0]
        self.routes = {
            f"/repos/{REPO}/issues/3": (200, issue_3, {}),
            f"/repos/{REPO}/issues/2": (200, load("issues_page1.json")[1], {}),
            f"/repos/{REPO}/issues/3/comments": (200, load("comments_3.json"), {}),
            f"/repos/{REPO}/labels": (200, load("labels.json"), {}),
            f"/repos/{REPO}/issues/404": (404, {"message": "Not Found"}, {}),
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST":
            body = json.loads(request.content)
            if path.endswith("/labels"):
                return httpx.Response(200, json=[{"name": n} for n in body["labels"]])
            return httpx.Response(201, json={"id": 99, "body": body["body"], "user": {"login": "bot"}})
        if path == f"/repos/{REPO}/issues":
            if request.url.params.get("page") == "2":
                return httpx.Response(200, json=load("issues_page2.json"))
            nxt = f'<{BASE}/repos/{REPO}/issues?state=open&per_page=100&page=2>; rel="next"'
            return httpx.Response(200, json=load("issues_page1.json"), headers={"Link": nxt})
        status, body, headers = self.routes.get(path, (404, {"message": "Not Found"}, {}))
        return httpx.Response(status, json=body, headers=headers)


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub()


@pytest.fixture
def gh(fake: FakeGitHub) -> GitHubClient:
    http = httpx.Client(transport=httpx.MockTransport(fake.handler))
    return GitHubClient("TOKEN", allowed_repos=(REPO,), base_url=BASE, http=http)
