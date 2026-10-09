"""The structured result of triaging one issue, submitted through the finish tool."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from triagebot.tools.registry import Tool


class Decision(BaseModel):
    labels: list[str] = Field(
        default_factory=list, description="Existing repository labels that apply to this issue"
    )
    duplicate_of: int = Field(
        default=0, ge=0, description="Number of the earlier issue this duplicates, or 0 if none"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Questions for the author if information is missing; empty if nothing is",
    )
    comment: str = Field(
        default="", max_length=2000,
        description="Short comment to post for the author, or an empty string for no comment",
    )
    reasoning: str = Field(
        min_length=1, max_length=1000,
        description="Why, in one to three sentences, citing evidence such as issue numbers",
    )


FINISH = Tool(
    "finish",
    "Submit your final triage decision. Call this exactly once, when you are done.",
    Decision,
    func=lambda decision: decision,
).spec()


def validate_decision(
    args: dict, issue_number: int, repo_labels: set[str]
) -> tuple[Decision | None, str]:
    """Check the decision's shape and its facts. Returns (decision, "") or (None, error)."""
    try:
        decision = Decision.model_validate(args)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return None, f"Invalid arguments for finish: {problems}"
    unknown = [lb for lb in decision.labels if lb not in repo_labels]
    if unknown:
        return None, (
            f"Unknown labels: {', '.join(unknown)}. "
            f"Use only existing labels: {', '.join(sorted(repo_labels))}"
        )
    if decision.duplicate_of == issue_number:
        return None, "An issue cannot be a duplicate of itself. Use 0 if it is not a duplicate."
    return decision, ""
