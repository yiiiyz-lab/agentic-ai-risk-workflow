
import pandas as pd
import pytest

from src.tools.get_market_context import get_market_context


@pytest.fixture
def sample_files(tmp_path):
    dates = pd.bdate_range("2023-03-01", periods=12)

    prices = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "SPY": [100 + i for i in range(12)],
        "IEF": [100 - i * 0.5 for i in range(12)],
    })

    articles = pd.DataFrame([
        {
            "event_id": "A1",
            "publication_date": "2023-03-10",
            "source": "Source A",
            "title": "Banking event",
            "url": "https://example.org/a1",
            "summary": "Historical event summary",
            "tags": "banking_stress;credit",
        },
        {
            "event_id": "A2",
            "publication_date": "2023-03-12",
            "source": "Source B",
            "title": "Policy response",
            "url": "https://example.org/a2",
            "summary": "Historical policy summary",
            "tags": "banking_stress;rates",
        },
        {
            "event_id": "A3",
            "publication_date": "2023-03-22",
            "source": "Source C",
            "title": "Future event",
            "url": "https://example.org/a3",
            "summary": "Future event summary",
            "tags": "rates",
        },
    ])

    p = tmp_path / "prices.csv"
    a = tmp_path / "articles.csv"
    prices.to_csv(p, index=False)
    articles.to_csv(a, index=False)
    return p, a, dates


def run(files, date, **kwargs):
    p, a, _ = files
    return get_market_context(
        portfolio_id="TEST",
        as_of_date=date,
        prices_path=p,
        articles_path=a,
        **kwargs,
    )


def test_market_returns(sample_files):
    _, _, dates = sample_files
    result = run(
        sample_files, dates[5], windows=(5,)
    )
    assert result["market_performance"]["5"][
        "etf_returns"
    ]["SPY"] == pytest.approx(0.05)


def test_future_articles_excluded(sample_files):
    result = run(sample_files, "2023-03-15")
    ids = [
        a["event_id"]
        for a in result["external_context"]["articles"]
    ]
    assert "A1" in ids
    assert "A2" in ids
    assert "A3" not in ids


def test_same_day_articles_excluded(sample_files):
    result = run(sample_files, "2023-03-10")
    assert result["external_context"]["article_count"] == 0


def test_no_evidence_status(sample_files):
    result = run(
        sample_files,
        "2023-03-08",
        event_lookback_days=2,
    )
    assert result["external_context"]["status"] == (
        "NOT_AVAILABLE"
    )


def test_topic_filter(sample_files):
    result = run(
        sample_files,
        "2023-03-15",
        topic_tags=["rates"],
    )
    ids = [
        a["event_id"]
        for a in result["external_context"]["articles"]
    ]
    assert ids == ["A2"]


def test_short_history(sample_files):
    _, _, dates = sample_files
    result = run(
        sample_files, dates[2], windows=(20,)
    )
    assert result["market_performance"]["20"][
        "status"
    ] == "NOT_EVALUATED"


def test_missing_article_file(sample_files, tmp_path):
    p, _, dates = sample_files
    with pytest.raises(FileNotFoundError):
        get_market_context(
            "TEST",
            dates[5],
            prices_path=p,
            articles_path=tmp_path / "missing.csv",
        )


def test_duplicate_article_id(sample_files):
    p, a, dates = sample_files
    df = pd.read_csv(a)
    df.loc[1, "event_id"] = "A1"
    df.to_csv(a, index=False)

    with pytest.raises(ValueError):
        run(sample_files, dates[5])


def test_invalid_url(sample_files):
    p, a, dates = sample_files
    df = pd.read_csv(a)
    df.loc[0, "url"] = "not-a-url"
    df.to_csv(a, index=False)

    with pytest.raises(ValueError):
        run(sample_files, dates[5])


def test_no_future_prices(sample_files):
    _, _, dates = sample_files
    result = run(
        sample_files,
        dates[5],
        windows=(5,),
    )
    assert result["market_performance"]["5"][
        "end_date"
    ] == dates[5].strftime("%Y-%m-%d")


def test_missing_price_snapshot(sample_files):
    with pytest.raises(ValueError):
        run(sample_files, "2023-03-11")
