import datetime
import hashlib
import re
from pathlib import Path

import pandas as pd
from mini_transcript_search import Criteria, ModelHandler
from mini_transcript_search.search import SearchResult
from pydantic_store import PydanticDBM

SEARCH_CRITERIA = Criteria(
    [
        "register of members interests",
        "May I draw attention to my interests as registered in the Register of Members Financial Interests",
        "May I draw attention to my entry in the Register of Members' Financial Interests?",
        "May I draw attention to my interests in register?",
        "May I draw attention to my interests as declared in the register?",
        "I refer Members to my registered interest.",
    ],
    score_type="nearest",
)

MUST_CONTAIN = ["declar", "interest", "register"]


def get_handler(storage_path: Path) -> ModelHandler:
    return ModelHandler(
        use_local_model=False,
        override_stored=False,
        storage_path=storage_path,
        silent=True,
    )


def bold_icase(text: str, word: str) -> str:
    """
    Bold a word in a case insensitive way.
    We are passing 'interest' but want to bold 'Interest' for example.
    """
    return re.sub(re.compile(f"({word})", re.IGNORECASE), r"**\1**", text)


def bold_text(text: str, words: list[str] | None = None) -> str:
    if words is None:
        words = MUST_CONTAIN
    bolded_text = text
    for m in words:
        bolded_text = bold_icase(bolded_text, m)
    return bolded_text


def _search_cache_key(
    start_date: datetime.date,
    end_date: datetime.date,
    threshold: float,
) -> str:
    """Generate a cache key for a search query based on its parameters."""
    raw = f"{start_date}:{end_date}:{threshold}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def search_last_month(
    handler: ModelHandler,
    threshold: float = 0.35,
) -> pd.DataFrame:
    """
    Run vector search for register-of-interest declarations in the last month.
    Returns a filtered, deduplicated DataFrame ready for display.
    Caches the raw SearchResult so repeat runs on the same day are instant.
    """
    today = datetime.date.today()
    one_month_ago = today - datetime.timedelta(days=40)

    cache_dir = Path("data", "regmem_vector_search")
    cache_dir.mkdir(parents=True, exist_ok=True)
    store: PydanticDBM[SearchResult] = PydanticDBM(
        cache_dir / "search_results.sqlite",
        storage_format=SearchResult,
    )

    cache_key = _search_cache_key(one_month_ago, today, threshold) + "para"
    results = store.get(cache_key)

    if results is None:
        last_month = ModelHandler.DateRange(start_date=one_month_ago, end_date=today)
        results = handler.query(
            SEARCH_CRITERIA,
            threshold=threshold,
            date_range=last_month,
            chamber=ModelHandler.Chamber.COMMONS,
            transcript_type=ModelHandler.TranscriptType.DEBATES,
            return_paragraph=True,
        )
        store[cache_key] = results

    store.close()

    df = results.df()
    return process_results_df(df)


def process_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns, filter by must-contain words, and deduplicate.
    """
    df["date"] = df["speech_id"].apply(lambda x: x.split("/")[2][:10])
    df["int_person_id"] = df["person_id"].apply(lambda x: x.split("/")[-1] if x else "")

    passes_text = df["matched_text"].apply(
        lambda text: any(word in text.lower() for word in MUST_CONTAIN)
    )
    df = df[passes_text]

    # dedupe on debate_url
    df = df.drop_duplicates(subset=["debate_url"])

    # sort by reverse date
    df = df.sort_values("date", ascending=False)

    return df
