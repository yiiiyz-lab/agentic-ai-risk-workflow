
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


VALID_STATUSES = {
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "BUDGET_EXCEEDED",
    "REQUIRES_REVIEW",
}


@dataclass
class InvestigationState:
    portfolio_id: str
    as_of_date: str
    exception_id: str
    rule_id: str
    exception_status: str

    investigation_id: str = field(
        default_factory=lambda: str(uuid4())
    )
    status: str = "PENDING"
    trace: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    limitations: list = field(default_factory=list)

    def __post_init__(self):
        if self.exception_status not in {
            "WARNING",
            "BREACH",
            "NOT_EVALUATED",
        }:
            raise ValueError(
                "Investigation requires a non-PASS exception"
            )

        if not all([
            self.portfolio_id,
            self.as_of_date,
            self.exception_id,
            self.rule_id,
        ]):
            raise ValueError(
                "Missing investigation identifiers"
            )

        self.as_of_date = datetime.strptime(
            self.as_of_date, "%Y-%m-%d"
        ).strftime("%Y-%m-%d")

    def set_status(self, status):
        if status not in VALID_STATUSES:
            raise ValueError("Invalid investigation status")

        self.status = status
        self.add_trace(
            "STATUS_CHANGE",
            {"status": status},
        )

    def add_trace(self, event_type, details):
        self.trace.append({
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "event_type": event_type,
            "details": details,
        })

    def add_evidence(self, tool_name, result):
        self.evidence.append({
            "tool_name": tool_name,
            "result": result,
        })

        self.add_trace(
            "EVIDENCE_RECORDED",
            {
                "tool_name": tool_name,
                "evidence_index": len(self.evidence) - 1,
            },
        )

    def add_finding(self, statement, evidence_indices):
        if not statement or not evidence_indices:
            raise ValueError(
                "Findings require a statement and evidence"
            )

        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(self.evidence)
            for index in evidence_indices
        ):
            raise ValueError(
                "Finding references invalid evidence"
            )

        self.findings.append({
            "statement": statement,
            "evidence_indices": evidence_indices,
            "review_status": "PENDING",
        })

    def to_dict(self):
        return {
            "investigation_id": self.investigation_id,
            "portfolio_id": self.portfolio_id,
            "as_of_date": self.as_of_date,
            "exception_id": self.exception_id,
            "rule_id": self.rule_id,
            "exception_status": self.exception_status,
            "status": self.status,
            "trace": self.trace,
            "evidence": self.evidence,
            "findings": self.findings,
            "limitations": self.limitations,
        }
