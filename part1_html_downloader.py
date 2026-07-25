part1_html_downloader.py
-------------------------
Downloads raw HTML for every URL listed in dataset/urls.csv and saves each
page as a .txt file inside Alloutputs/. Uses a thread pool for concurrency
since HTML downloading is I/O-bound.

Input : dataset/urls.csv          (columns: url, label)
Output: Alloutputs/<hash>.txt     (one file per URL)
        Alloutputs/download_manifest.csv  (url, label, filename, status)
"""

import os
import csv
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import (
    DATASET_CSV,
    HTML_OUTPUT_DIR,
    HTML_REQUEST_TIMEOUT,
    HTML_MAX_WORKERS,
    HTML_RETRY_COUNT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def url_to_filename(url: str) -> str:
    """Deterministic filename for a URL using an md5 hash (avoids illegal path chars)."""
    return hashlib.md5(url.encode("utf-8")).hexdigest() + ".txt"


def download_html(url: str) -> tuple[bool, str]:
    """Attempt to download HTML for a single URL, with retries. Returns (success, html_or_error)."""
    last_error = ""
    for attempt in range(1, HTML_RETRY_COUNT + 2):
        try:
            resp = requests.get(
                url, headers=HEADERS, timeout=HTML_REQUEST_TIMEOUT, verify=False
            )
            resp.raise_for_status()
            return True, resp.text
        except Exception as exc:  # noqa: BLE001 - want to catch & log any network failure
            last_error = str(exc)
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, HTML_RETRY_COUNT + 1, url, exc)
    return False, last_error


def process_row(row: dict) -> dict:
    url = row["url"].strip()
    label = row.get("label", "unknown").strip()
    filename = url_to_filename(url)
    filepath = os.path.join(HTML_OUTPUT_DIR, filename)

    success, content = download_html(url)
    if success:
        with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
            f.write(content)
        status = "success"
    else:
        status = f"failed: {content}"
        filename = ""

    return {"url": url, "label": label, "filename": filename, "status": status}


def load_urls(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Create dataset/urls.csv with columns: url,label"
        )
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    os.makedirs(HTML_OUTPUT_DIR, exist_ok=True)
    rows = load_urls(DATASET_CSV)
    logger.info("Loaded %d URLs from %s", len(rows), DATASET_CSV)

    results = []
    with ThreadPoolExecutor(max_workers=HTML_MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): row for row in rows}
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if i % 50 == 0 or i == len(rows):
                logger.info("Progress: %d/%d", i, len(rows))

    manifest_path = os.path.join(HTML_OUTPUT_DIR, "download_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "filename", "status"])
        writer.writeheader()
        writer.writerows(results)

    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(
        "Done. %d/%d downloaded successfully. Manifest saved to %s",
        success_count, len(results), manifest_path,
    )


if __name__ == "__main__":
    main()
