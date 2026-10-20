
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.tools.common import normalize_date, read_csv


DEFAULT_PRICES = "data/sample/adjusted_prices.csv"
DEFAULT_ARTICLES = "data/sample/market_context_articles.csv"


def get_market_context(
    portfolio_id,
    as_of_date,
    windows=(5, 20, 63),
    event_lookback_days=30,
    topic_tags=None,
    prices_path=DEFAULT_PRICES,
    articles_path=DEFAULT_ARTICLES,
):
    """
    Return historical ETF performance and curated external evidence.

    Publication dates must be strictly earlier than as_of_date
    because the dataset has day-level, not intraday, timestamps.

    No causal inference or live news retrieval is performed.
    """
    if not portfolio_id or not isinstance(portfolio_id, str):
        raise ValueError("portfolio_id must be a non-empty string")

    date = pd.Timestamp(as_of_date)
    if pd.isna(date):
        raise ValueError("Invalid as_of_date")
    date = date.normalize()

    if not windows or any(
        not isinstance(w, int) or isinstance(w, bool) or w <= 0
        for w in windows
    ):
        raise ValueError("windows must contain positive integers")

    if not isinstance(event_lookback_days, int) or (
        event_lookback_days < 0
    ):
        raise ValueError("event_lookback_days must be non-negative")

    if topic_tags is not None:
        if isinstance(topic_tags, str):
            topic_tags = [topic_tags]
        if not all(
            isinstance(t, str) and t.strip() for t in topic_tags
        ):
            raise ValueError("Invalid topic_tags")
        topic_tags = set(topic_tags)

    prices = read_csv(prices_path, required_columns=["date"])
    prices["date"] = pd.to_datetime(
        prices["date"], errors="raise"
    ).dt.normalize()

    if prices["date"].duplicated().any():
        raise ValueError("Duplicate price dates")

    prices = prices.loc[
        prices["date"] <= date
    ].sort_values("date").reset_index(drop=True)

    if prices.empty or date not in set(prices["date"]):
        raise ValueError("No exact market price snapshot for as_of_date")

    symbols = [c for c in prices.columns if c != "date"]
    if not symbols:
        raise ValueError("No ETF price columns")

    for symbol in symbols:
        prices[symbol] = pd.to_numeric(
            prices[symbol], errors="coerce"
        )

    current = prices.iloc[-1]
    performance = {}

    for window in sorted(set(windows)):
        if len(prices) <= window:
            performance[str(window)] = {
                "status": "NOT_EVALUATED",
                "reason": "INSUFFICIENT_HISTORY",
            }
            continue

        prior = prices.iloc[-1 - window]
        returns = {}

        for symbol in symbols:
            p0, p1 = prior[symbol], current[symbol]

            if (
                not np.isfinite(p0)
                or not np.isfinite(p1)
                or p0 <= 0
                or p1 <= 0
            ):
                returns[symbol] = None
            else:
                returns[symbol] = float(p1 / p0 - 1)

        performance[str(window)] = {
            "status": (
                "PASS"
                if all(v is not None for v in returns.values())
                else "NOT_EVALUATED"
            ),
            "start_date": normalize_date(prior["date"]),
            "end_date": normalize_date(date),
            "etf_returns": returns,
        }

    articles = read_csv(
        articles_path,
        required_columns=[
            "event_id", "publication_date", "source",
            "title", "url", "summary", "tags",
        ],
    )

    if articles["event_id"].isna().any() or (
        articles["event_id"].duplicated().any()
    ):
        raise ValueError("Missing or duplicate article event_id")

    articles["publication_date"] = pd.to_datetime(
        articles["publication_date"],
        format="%Y-%m-%d",
        errors="raise",
    ).dt.normalize()

    if articles["publication_date"].isna().any():
        raise ValueError("Missing article publication date")

    for field in ("source", "title", "url", "summary"):
        if articles[field].isna().any() or (
            articles[field].astype(str).str.strip() == ""
        ).any():
            raise ValueError(f"Missing article {field}")

    for url in articles["url"]:
        parsed = urlparse(str(url))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Invalid article URL")

    start = date - pd.Timedelta(days=event_lookback_days)

    # Strictly earlier than the investigation date:
    # day-level metadata cannot prove same-day availability.
    eligible = articles.loc[
        (articles["publication_date"] >= start)
        & (articles["publication_date"] < date)
    ].copy()

    if topic_tags is not None:
        eligible = eligible.loc[
            eligible["tags"].fillna("").apply(
                lambda value: bool(
                    set(str(value).split(";")) & topic_tags
                )
            )
        ]

    eligible = eligible.sort_values(
        ["publication_date", "event_id"],
        ascending=[False, True],
    )

    evidence = []

    for _, article in eligible.iterrows():
        evidence.append({
            "event_id": str(article["event_id"]),
            "publication_date": normalize_date(
                article["publication_date"]
            ),
            "source": str(article["source"]),
            "title": str(article["title"]),
            "url": str(article["url"]),
            "summary": str(article["summary"]),
            "tags": [
                tag.strip()
                for tag in str(article["tags"]).split(";")
                if tag.strip()
            ] if pd.notna(article["tags"]) else [],
            "evidence_type": "CURATED_HISTORICAL_SOURCE",
            "verification_status": "CURATED_SOURCE",
        })

    return {
        "tool": "get_market_context",
        "portfolio_id": portfolio_id,
        "as_of_date": normalize_date(date),
        "market_performance": performance,
        "external_context": {
            "status": (
                "AVAILABLE" if evidence else "NOT_AVAILABLE"
            ),
            "article_count": len(evidence),
            "articles": evidence,
        },
        "data_sources": {
            "prices": str(Path(prices_path)),
            "articles": str(Path(articles_path)),
        },
        "methodology": {
            "returns": "Adjusted price end/start - 1",
            "windows": "Trading observations",
            "article_window": "Calendar days",
            "publication_cutoff": "Strictly before as_of_date",
            "topic_filter": (
                "Exact match against semicolon-separated tags"
            ),
        },
        "limitations": [
            "The curated article dataset is not comprehensive.",
            "Source URLs are stored but not reverified at runtime.",
            "Publication dates are day-level only.",
            "Historical publication date does not establish "
            "intraday availability.",
            "Market performance is descriptive, not causal.",
            "ETF returns are not portfolio returns.",
            "Portfolio ID is metadata; this tool does not "
            "validate portfolio holdings or calculate NAV.",
        ],
    }
