import json

import pytest
from conftest import REPO, FakeGitHub

from triagebot.github.client import GitHubClient, GitHubError, RepoNotAllowed


def test_list_open_issues_follows_pages_and_drops_pull_requests(gh: GitHubClient) -> None:
    issues = gh.list_open_issues(REPO)
    assert [i.number for i in issues] == [3, 1]


def test_get_issue_includes_comments(gh: GitHubClient) -> None:
    issue = gh.get_issue(REPO, 3)
    assert issue.title.startswith("App crashes")
    assert issue.labels == ["bug"]
    assert issue.comments[0].author == "maintainer"


def test_sends_auth_and_pinned_api_version(gh: GitHubClient, fake: FakeGitHub) -> None:
    gh.list_labels(REPO)
    headers = fake.requests[0].headers
    assert headers["Authorization"] == "Bearer TOKEN"
    assert headers["X-GitHub-Api-Version"] == "2022-11-28"


def test_allowlist_blocks_before_any_request(gh: GitHubClient, fake: FakeGitHub) -> None:
    with pytest.raises(RepoNotAllowed):
        gh.get_issue("someone/else", 1)
    assert fake.requests == []


def test_http_errors_keep_status_code(gh: GitHubClient) -> None:
    with pytest.raises(GitHubError) as err:
        gh.get_issue(REPO, 404)
    assert err.value.status_code == 404
    assert "Not Found" in str(err.value)


def test_list_issues_requests_all_states(gh: GitHubClient, fake: FakeGitHub) -> None:
    issues = gh.list_issues(REPO)
    assert fake.requests[0].url.params["state"] == "all"
    assert [i.number for i in issues] == [3, 1]


def test_write_calls_send_json_and_respect_allowlist(gh: GitHubClient, fake: FakeGitHub) -> None:
    assert gh.add_labels(REPO, 3, ["bug"]) == ["bug"]
    comment = gh.create_comment(REPO, 3, "hello")
    assert comment.id == 99
    post = fake.requests[-1]
    assert post.method == "POST" and json.loads(post.content) == {"body": "hello"}
    with pytest.raises(RepoNotAllowed):
        gh.create_comment("someone/else", 1, "hi")
    assert len(fake.requests) == 2
