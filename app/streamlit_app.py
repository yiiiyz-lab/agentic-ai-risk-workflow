
"""Interactive AI Risk Investigation showcase."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting.report_builder import build_report


EXCEPTIONS_PATH = ROOT / "data/outputs/risk_exceptions.csv"
INVESTIGATION_PATH = (
    ROOT / "data/outputs/live_investigation_equity_20260930.json"
)


st.set_page_config(
    page_title="AI Risk Investigator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_exceptions():
    if not EXCEPTIONS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(EXCEPTIONS_PATH)


@st.cache_data
def load_investigation():
    if not INVESTIGATION_PATH.exists():
        return None

    with INVESTIGATION_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def render_chart(chart):
    """Render supported evidence-backed chart specifications."""

    df = pd.DataFrame(chart.data)

    if df.empty:
        return

    fields = [
        field for field in chart.y_fields
        if field in df.columns
    ]

    if not fields or chart.x_field not in df.columns:
        st.info("Chart data is unavailable.")
        return

    if chart.chart_type == "pie":
        fig = px.pie(
            df,
            names=chart.x_field,
            values=fields[0],
            hole=0.45,
        )

    elif chart.chart_type == "line":
        fig = px.line(
            df,
            x=chart.x_field,
            y=fields,
            markers=True,
        )

    else:
        fig = px.bar(
            df,
            x=chart.x_field,
            y=fields,
            barmode="group",
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=15, r=15, t=20, b=15),
        legend_title_text="",
        height=350,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    if chart.notes:
        st.caption(chart.notes)

    st.caption(
        f"Evidence references: {chart.evidence_indices}"
    )


def render_report(report, investigation):
    """Render the structured AI investigation report."""

    st.title("AI Investigation Report")

    st.caption(
        f"{report.portfolio_id} | {report.as_of_date} "
        f"| {report.rule_id}"
    )

    st.warning(
        "AI-generated investigation draft. "
        "Human review and formal report validation are pending."
    )

    tab_summary, tab_findings, tab_evidence, tab_trace = st.tabs(
        [
            "Overview & Charts",
            "AI Findings",
            "Supporting Evidence",
            "Investigation Trace",
        ]
    )

    with tab_summary:
        st.subheader("Executive Summary")
        st.markdown(report.executive_summary)

        st.subheader("Risk Analysis")

        if report.visualizations:
            for start in range(0, len(report.visualizations), 2):
                columns = st.columns(2)

                for offset, column in enumerate(columns):
                    index = start + offset

                    if index >= len(report.visualizations):
                        continue

                    chart = report.visualizations[index]

                    with column:
                        with st.container(border=True):
                            st.markdown(f"**{chart.title}**")
                            render_chart(chart)
        else:
            st.info("No evidence-backed charts are available.")

        st.subheader("Analytical Tables")

        for table in report.tables:
            with st.expander(table.title, expanded=False):
                st.dataframe(
                    pd.DataFrame(table.rows),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    f"Evidence references: {table.evidence_indices}"
                )

    with tab_findings:
        st.subheader("AI Investigation Findings")

        for finding in report.findings:
            with st.container(border=True):
                st.markdown(f"#### {finding.title}")
                st.caption(
                    f"Classification: {finding.classification}"
                )
                st.markdown(finding.narrative)

                if finding.evidence_indices:
                    st.caption(
                        "Evidence: "
                        + ", ".join(
                            str(i) for i in finding.evidence_indices
                        )
                    )
                else:
                    st.caption(
                        "No explicit evidence citation in this section."
                    )

        st.subheader("Limitations")

        if report.limitations:
            for limitation in report.limitations:
                st.write(f"• {limitation}")
        else:
            st.info("No limitations recorded.")

    with tab_evidence:
        st.subheader("Evidence & Supporting Analysis")

        evidence = investigation.get("evidence", [])

        for index, entry in enumerate(evidence):
            name = entry.get("tool_name", "Unknown tool")

            with st.expander(
                f"Evidence {index}: {name}"
            ):
                st.json(entry.get("result", {}))

        st.subheader("Supporting Tables")

        for table in report.tables:
            st.markdown(f"**{table.title}**")
            st.dataframe(
                pd.DataFrame(table.rows),
                width="stretch",
                hide_index=True,
            )

    with tab_trace:
        st.subheader("AI Investigation Trace")

        trace = investigation.get("trace", [])

        for index, event in enumerate(trace):
            event_type = event.get("event_type", "UNKNOWN")
            timestamp = event.get("timestamp_utc", "")

            with st.expander(
                f"{index + 1}. {event_type} — {timestamp}"
            ):
                st.json(event.get("details", {}))

        st.subheader("Execution Summary")

        budget = investigation.get("budget", {})

        if budget:
            st.json(budget)

        st.subheader("Review Status")
        st.info(
            f"Report: {report.report_status} | "
            f"Review: {report.review_status}"
        )

    st.divider()

    st.download_button(
        "Download Structured Report (JSON)",
        data=json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        file_name=(
            f"investigation_{report.exception_id}.json"
        ),
        mime="application/json",
    )


def main():
    st.sidebar.title("AI Risk Investigator")
    st.sidebar.caption(
        "From risk exceptions to evidence-grounded AI investigations"
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "Exception Dashboard"

    page = st.sidebar.radio(
        "Navigation",
        ["Exception Dashboard", "Investigation Report"],
        key="page",
    )

    exceptions = load_exceptions()
    investigation = load_investigation()

    if page == "Exception Dashboard":
        st.title("Portfolio Risk Monitoring")
        st.caption(
            "Explore detected risk exceptions and available "
            "AI investigations."
        )

        if exceptions.empty:
            st.warning(
                "No exception data found. "
                "Run the risk exception detector first."
            )
            return

        required = {
            "portfolio_id", "date", "rule_id",
            "status", "exception_id",
        }

        if not required.issubset(exceptions.columns):
            st.error(
                "The exception file is missing required columns."
            )
            return

        portfolios = sorted(
            exceptions["portfolio_id"].dropna().unique()
        )

        portfolio = st.selectbox(
            "Portfolio",
            portfolios,
        )

        portfolio_rows = exceptions[
            exceptions["portfolio_id"] == portfolio
        ].copy()

        dates = sorted(
            portfolio_rows["date"].dropna().astype(str).unique(),
            reverse=True,
        )

        selected_date = st.selectbox(
            "As-of Date",
            dates,
        )

        filtered = portfolio_rows[
            portfolio_rows["date"].astype(str) == selected_date
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric("Detected Exceptions", len(filtered))
        col2.metric(
            "Breaches",
            int((filtered["status"] == "BREACH").sum()),
        )
        col3.metric(
            "Warnings",
            int((filtered["status"] == "WARNING").sum()),
        )

        st.subheader("Detected Exceptions")

        for _, exception in filtered.iterrows():
            exception_id = str(exception["exception_id"])

            with st.container(border=True):
                left, right = st.columns([4, 1])

                with left:
                    st.markdown(
                        f"**{exception['rule_id']}**"
                    )
                    st.caption(
                        f"{exception['status']} | {exception_id}"
                    )

                with right:
                    available = (
                        investigation is not None
                        and investigation.get("exception_id")
                        == exception_id
                        and investigation.get("portfolio_id")
                        == portfolio
                        and investigation.get("as_of_date")
                        == selected_date
                    )

                    if st.button(
                        "View Investigation",
                        key=exception_id,
                        disabled=not available,
                    ):
                        st.session_state["selected_exception"] = (
                            exception_id
                        )
                        st.session_state["page"] = (
                            "Investigation Report"
                        )
                        st.rerun()

                    if not available:
                        st.caption("Demo report unavailable")

    else:
        st.title("Investigation Explorer")

        if investigation is None:
            st.warning(
                "No saved investigation found."
            )
            return

        report = build_report(investigation)

        render_report(report, investigation)


if __name__ == "__main__":
    main()
