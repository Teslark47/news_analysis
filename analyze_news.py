""" Ali Zandieh - Interview task
Persian news EDA with explicit input-file selection.
Run from the project folder: python analyze_news.py
Expected input files in data/: main_news.csv, categories.csv, category_news.csv.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
DATA_DIR = ROOT / CFG.get("data_dir", "data")
OUT_DIR = ROOT / CFG.get("output_dir", "outputs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = set("""
و یا از به در با برای که را این آن یک تا است بود شد شده می شود هستند هست نیست
های ها آن‌ها آنها ایشان ما شما من تو او هم نیز اگر اما ولی چون زیرا بر پس پیش
زیر روی بین هر همه هیچ چند بعضی دیگر همین همان چنین چطور چرا کجا چه کی چگونه
وقتی بعد قبل اکنون امروز دیروز فردا همچنین بنابراین لذا ضمن مورد موارد درباره
علیه بدون همراه توسط طی جهت اساس حال حاضر گذشته خواهد خواهند کرده کردن کرد کند
کنند باشد باشند بودن داشتن دارد دارند داشت داشتند خبر اخبار گزارش اعلام گفت گفته
افزود اظهار تهران ایران کشور مردم منبع عکس تصویر اختصاصی خبرنگار خبرنگاران
""".split())

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def log(message: str) -> None:
    print(f"[INFO] {message}")


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV with common encodings/separators; never pass low_memory to python engine."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}\nPlace it in the project's data folder.")
    last_error = None
    attempts = [(enc, sep) for enc in ("utf-8-sig", "utf-8", "cp1256") for sep in (None, ",", ";", "\t")]
    for enc, sep in attempts:
        try:
            frame = pd.read_csv(path, encoding=enc, sep=sep, engine="python", on_bad_lines="warn")
            # Reject a likely bad parse where a normal CSV was read as a single column.
            if len(frame.columns) == 1 and path.stat().st_size > 100:
                continue
            log(f"Loaded {path.name}: {len(frame):,} rows x {len(frame.columns)} columns (encoding={enc}, sep={sep!r})")
            return frame
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path.name}: {last_error}")


def norm_col(value: object) -> str:
    return re.sub(r"[\W_]+", "", str(value).strip().lower(), flags=re.UNICODE)


def choose_column(columns, candidates):
    lookup = {norm_col(c): c for c in columns}
    for candidate in candidates:
        if norm_col(candidate) in lookup:
            return lookup[norm_col(candidate)]
    for col in columns:
        normalized = norm_col(col)
        for candidate in candidates:
            cand = norm_col(candidate)
            if len(cand) >= 3 and (cand in normalized or normalized in cand):
                return col
    return None


def normalize_persian(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).translate(PERSIAN_DIGITS)
    text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    # Keep the half-space as a separator for this simple word-count analysis.
    text = text.replace("\u200c", " ").replace("\u200f", "").replace("\u200e", "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def save_bar(data, label_col, value_col, title, filename, horizontal=False, limit=20):
    if data is None or data.empty:
        return
    subset = data.head(limit).copy()
    if horizontal:
        fig, ax = plt.subplots(figsize=(11, max(5, len(subset) * 0.32)))
        ax.barh(subset[label_col].astype(str)[::-1], subset[value_col][::-1])
        ax.set_xlabel(value_col)
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(subset[label_col].astype(str), subset[value_col])
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def main():
    # IMPORTANT: choose the news table by its exact filename; never the first *.csv.
    news_path = DATA_DIR / "main_news.csv"
    categories_path = DATA_DIR / "categories.csv"
    relationships_path = DATA_DIR / "category_news.csv"

    log(f"Project root: {ROOT}")
    log(f"Data folder: {DATA_DIR}")
    log(f"Output folder: {OUT_DIR}")
    news = read_csv_safely(news_path)

    categories = read_csv_safely(categories_path) if categories_path.exists() else None
    category_links = read_csv_safely(relationships_path) if relationships_path.exists() else None

    title_col = choose_column(news.columns, ["title", "headline", "عنوان", "تیتر"])
    body_col = choose_column(news.columns, ["description", "content", "text", "body", "news", "متن", "شرح", "خلاصه"])
    category_col = choose_column(news.columns, ["category", "categories", "cat", "دسته", "دسته بندی", "موضوع"])
    source_col = choose_column(news.columns, ["source", "agency", "publisher", "website", "خبرگزاری", "منبع", "site"])
    date_col = choose_column(news.columns, ["date", "published_at", "published", "created_at", "pubdate", "تاریخ", "زمان"])
    url_col = choose_column(news.columns, ["url", "link", "news_url", "لینک"])
    news_id_col = choose_column(news.columns, ["news_id", "article_id", "id_news", "id خبر", "شناسه خبر"])

    work = news.copy()
    category_method = ""

    # If categories are already in the news table, use that column.
    if category_col:
        work["category_normalized"] = work[category_col].fillna("نامشخص").astype(str).str.strip().replace("", "نامشخص")
        category_method = f"category column '{category_col}' in main_news.csv"
    # Join only on an explicit news ID column. Do NOT assume Unnamed: 0 is a news ID.
    elif categories is not None and category_links is not None and news_id_col:
        link_news_id = choose_column(category_links.columns, ["news_id"])
        link_cat_id = choose_column(category_links.columns, ["category_id", "cat_id"])
        cat_id = choose_column(categories.columns, ["id", "category_id", "cat_id"])
        cat_name = choose_column(categories.columns, ["readable", "name", "title", "نام", "دسته"])
        if all([link_news_id, link_cat_id, cat_id, cat_name]):
            cat_map = categories[[cat_id, cat_name]].drop_duplicates(subset=[cat_id]).copy()
            links = category_links[[link_news_id, link_cat_id]].dropna().merge(
                cat_map, left_on=link_cat_id, right_on=cat_id, how="left", validate="many_to_one"
            )
            # Keep one row per news ID for this single-label summary; multi-label detail is saved separately.
            links["_category_name"] = links[cat_name].fillna("نامشخص").astype(str)
            names_by_news = links.groupby(link_news_id)["_category_name"].agg(lambda s: " | ".join(sorted(set(s))))
            work["category_normalized"] = work[news_id_col].map(names_by_news).fillna("نامشخص")
            category_method = f"joined using explicit news ID column '{news_id_col}'"
    if "category_normalized" not in work:
        work["category_normalized"] = "نامشخص"
        category_method = "unavailable: no explicit news ID column in main_news.csv; no unsafe join performed"

    if title_col:
        work["title_normalized"] = work[title_col].map(normalize_persian)
    else:
        work["title_normalized"] = ""
    if body_col and body_col != title_col:
        work["body_normalized"] = work[body_col].map(normalize_persian)
    elif body_col:
        work["body_normalized"] = work["title_normalized"]
    else:
        work["body_normalized"] = ""
    work["analysis_text"] = (work["title_normalized"].fillna("") + " " + work["body_normalized"].fillna("")).str.strip()
    work["word_count"] = work["analysis_text"].map(lambda value: len(value.split()) if value else 0)

    # Quality report for the main table and any supporting tables.
    quality = {
        "input_file": news_path.name,
        "rows": int(news.shape[0]),
        "columns_count": int(news.shape[1]),
        "columns": [str(c) for c in news.columns],
        "duplicate_rows": int(news.duplicated().sum()),
        "detected_columns": {"title": title_col, "body": body_col, "category": category_col,
                             "source": source_col, "date": date_col, "url": url_col, "news_id": news_id_col},
        "category_method": category_method,
        "missing_by_column": {str(c): int(news[c].isna().sum()) for c in news.columns},
        "supporting_files": {
            "categories.csv": {"loaded": categories is not None, "rows": int(len(categories)) if categories is not None else None},
            "category_news.csv": {"loaded": category_links is not None, "rows": int(len(category_links)) if category_links is not None else None},
        },
    }
    (OUT_DIR / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    profile = pd.DataFrame({
        "column": [str(c) for c in news.columns],
        "missing_count": [int(news[c].isna().sum()) for c in news.columns],
        "missing_percent": [round(float(news[c].isna().mean() * 100), 2) for c in news.columns],
        "dtype": [str(news[c].dtype) for c in news.columns],
        "unique_values": [int(news[c].nunique(dropna=True)) for c in news.columns],
    }).sort_values("missing_percent", ascending=False)
    profile.to_csv(OUT_DIR / "column_profile.csv", index=False, encoding="utf-8-sig")

    cat_stats = pd.DataFrame()
    if category_method.startswith("category column") or category_method.startswith("joined"):
        cat_stats = work["category_normalized"].value_counts(dropna=False).rename_axis("category").reset_index(name="count")
        cat_stats["percent"] = (cat_stats["count"] / max(len(work), 1) * 100).round(2)
        cat_stats.to_csv(OUT_DIR / "category_statistics.csv", index=False, encoding="utf-8-sig")
        save_bar(cat_stats, "category", "count", "News count by category", "01_category_counts.png", True)
    else:
        (OUT_DIR / "category_statistics_unavailable.txt").write_text(
            "Category statistics were not produced because main_news.csv has no explicit news ID column. "
            "Do not join category_news.news_id to Unnamed: 0 unless the dataset documentation confirms they are the same key.\n",
            encoding="utf-8")

    if source_col:
        src = work[source_col].fillna("نامشخص").astype(str).str.strip().replace("", "نامشخص")
        src_stats = src.value_counts().rename_axis("source").reset_index(name="count")
        src_stats["percent"] = (src_stats["count"] / max(len(work), 1) * 100).round(2)
        src_stats.to_csv(OUT_DIR / "source_statistics.csv", index=False, encoding="utf-8-sig")
        save_bar(src_stats, "source", "count", "News count by source", "02_source_counts.png", True)

    if date_col:
        dates = pd.to_datetime(work[date_col], errors="coerce", utc=True)
        valid_dates = dates.notna()
        if valid_dates.any():
            monthly = dates[valid_dates].dt.tz_convert(None).dt.to_period("M").astype(str).value_counts().sort_index()
            monthly_df = monthly.rename_axis("month").reset_index(name="count")
            monthly_df.to_csv(OUT_DIR / "monthly_statistics.csv", index=False, encoding="utf-8-sig")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(monthly_df["month"], monthly_df["count"], marker="o")
            ax.set_title("News distribution by month")
            ax.set_xlabel("Month")
            ax.set_ylabel("News count")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "04_monthly_distribution.png", dpi=160)
            plt.close(fig)

    tokens = [token for text in work["analysis_text"] for token in text.split()
              if len(token) > 1 and token not in STOPWORDS]
    top_words = Counter(tokens).most_common(20)
    top_df = pd.DataFrame(top_words, columns=["word", "frequency"])
    top_df.to_csv(OUT_DIR / "top_20_words.csv", index=False, encoding="utf-8-sig")
    save_bar(top_df, "word", "frequency", "Top 20 words after stopword removal", "03_top_words.png", True)

    nonzero = work.loc[work["word_count"] > 0, "word_count"]
    text_stats = {
        "articles_with_text": int((work["word_count"] > 0).sum()),
        "empty_texts": int((work["word_count"] == 0).sum()),
        "mean_words": round(float(nonzero.mean()), 2) if len(nonzero) else None,
        "median_words": float(nonzero.median()) if len(nonzero) else None,
        "min_words": int(nonzero.min()) if len(nonzero) else None,
        "max_words": int(nonzero.max()) if len(nonzero) else None,
    }
    (OUT_DIR / "text_statistics.json").write_text(json.dumps(text_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    work.to_csv(OUT_DIR / "cleaned_news.csv", index=False, encoding="utf-8-sig")

    report = [
        "# گزارش تحلیل دیتاست اخبار فارسی", "", "## نمای کلی",
        f"- فایل اصلی: `{news_path.name}`", f"- تعداد خبرها: **{len(news):,}**",
        f"- تعداد ستون‌ها: **{len(news.columns)}**", f"- ردیف‌های کاملاً تکراری: **{news.duplicated().sum():,}**",
        f"- ستون عنوان: `{title_col or 'یافت نشد'}`", f"- ستون متن: `{body_col or 'یافت نشد'}`",
        f"- ستون منبع: `{source_col or 'یافت نشد'}`", f"- وضعیت دسته‌بندی: {category_method}", "",
        "## کیفیت داده", "جزئیات ستون‌ها و مقادیر خالی در `column_profile.csv` و `data_quality.json` ثبت شده است.",
        "## پردازش متن", f"- خبرهای دارای متن: **{text_stats['articles_with_text']:,}**",
        f"- متن‌های خالی پس از پاک‌سازی: **{text_stats['empty_texts']:,}**",
        f"- میانگین تعداد واژه‌ها: **{text_stats['mean_words']}**",
        f"- میانه تعداد واژه‌ها: **{text_stats['median_words']}**",
        "- واژه‌های پرتکرار در `top_20_words.csv` ذخیره شده‌اند.", "",
        "## محدودیت‌ها",
        "- اگر شناسه صریح خبر در main_news.csv وجود نداشته باشد، آمار دسته‌بندی عمداً تولید نمی‌شود تا اتصال اشتباه رخ ندهد.",
        "- این نسخه تاریخ‌های عمومی قابل‌تفسیر توسط pandas را تحلیل می‌کند؛ تاریخ شمسی ممکن است به تبدیل جداگانه نیاز داشته باشد.",
        "- فهرست stopword ساده است و روی نتایج فراوانی واژه اثر می‌گذارد.",
        "- این پروژه تحلیل توصیفی و پیش‌پردازش مقدماتی انجام می‌دهد؛ مدل یادگیری ماشین یا تحلیل احساسات اجرا نمی‌کند.",
    ]
    if not cat_stats.empty:
        report += ["", "## توزیع دسته‌ها", f"- بزرگ‌ترین دسته در این داده: **{cat_stats.iloc[0]['category']}** ({int(cat_stats.iloc[0]['count']):,} خبر)."]
    (OUT_DIR / "report_fa.md").write_text("\n".join(report), encoding="utf-8")

    log(f"Analysis complete. Outputs are in: {OUT_DIR}")
    print(json.dumps({"input_file": news_path.name, "rows": len(news), "columns": list(news.columns),
                      "category_method": category_method, "text_statistics": text_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
