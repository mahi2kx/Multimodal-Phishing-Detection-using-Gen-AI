"""
part2_image_scraper.py
------------------------
Opens each URL with Playwright, extracts every image larger than the
configured minimum size, saves them to images/, then runs EasyOCR on each
saved image to pull out embedded text (a common phishing technique: baking
fake login text/branding into an image so text-based scrapers miss it).

Input : Alloutputs/download_manifest.csv  (produced by part1)
Output: images/<url_hash>_<n>.png
        ocr_output.csv   (url, label, image_path, ocr_text)
"""

import os
import csv
import logging
import asyncio
from urllib.parse import urljoin

from playwright.async_api import async_playwright
import easyocr

from config import (
    HTML_OUTPUT_DIR,
    IMAGES_OUTPUT_DIR,
    OCR_OUTPUT_CSV,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    PAGE_LOAD_TIMEOUT_MS,
    OCR_LANGUAGES,
    IMAGE_SCRAPER_CONCURRENCY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# EasyOCR reader is loaded once and reused (loading the model per-image is expensive)
_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Loading EasyOCR model (first run may download weights)...")
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=True)
    return _ocr_reader


def run_ocr(image_path: str) -> str:
    reader = get_ocr_reader()
    try:
        result = reader.readtext(image_path, detail=0)  # detail=0 -> just text strings
        return " ".join(result).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed for %s: %s", image_path, exc)
        return ""


async def scrape_images_for_url(context, url: str, url_hash: str, semaphore) -> list[str]:
    """Visit a URL, save qualifying images locally, return list of saved image paths."""
    saved_paths = []
    async with semaphore:
        page = await context.new_page()
        try:
            await page.goto(url, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)  # let lazy-loaded images settle

            img_elements = await page.query_selector_all("img")
            for idx, img in enumerate(img_elements):
                try:
                    box = await img.bounding_box()
                    if not box:
                        continue
                    if box["width"] < MIN_IMAGE_WIDTH or box["height"] < MIN_IMAGE_HEIGHT:
                        continue

                    src = await img.get_attribute("src")
                    if not src:
                        continue
                    full_src = urljoin(url, src)

                    filename = f"{url_hash}_{idx}.png"
                    filepath = os.path.join(IMAGES_OUTPUT_DIR, filename)

                    # Screenshot the element directly -> works for both <img> and background-rendered content
                    await img.screenshot(path=filepath)
                    saved_paths.append(filepath)
                except Exception as inner_exc:  # noqa: BLE001
                    logger.debug("Skipping image %d on %s: %s", idx, url, inner_exc)
                    continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", url, exc)
        finally:
            await page.close()
    return saved_paths


async def scrape_all(manifest_rows: list[dict]):
    os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)
    semaphore = asyncio.Semaphore(IMAGE_SCRAPER_CONCURRENCY)
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1366, "height": 900})

        tasks = []
        for row in manifest_rows:
            if row["status"] != "success":
                continue
            url_hash = os.path.splitext(row["filename"])[0]
            tasks.append(scrape_images_for_url(context, row["url"], url_hash, semaphore))

        all_image_lists = await asyncio.gather(*tasks)
        await browser.close()

    # Flatten (url, label, image_path) tuples for OCR stage
    valid_rows = [r for r in manifest_rows if r["status"] == "success"]
    for row, image_paths in zip(valid_rows, all_image_lists):
        for path in image_paths:
            results.append({"url": row["url"], "label": row["label"], "image_path": path})

    return results


def run_ocr_pipeline(image_rows: list[dict]):
    output_rows = []
    for i, row in enumerate(image_rows, start=1):
        text = run_ocr(row["image_path"])
        output_rows.append({**row, "ocr_text": text})
        if i % 25 == 0 or i == len(image_rows):
            logger.info("OCR progress: %d/%d", i, len(image_rows))

    with open(OCR_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "image_path", "ocr_text"])
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info("OCR results saved to %s", OCR_OUTPUT_CSV)


def load_manifest() -> list[dict]:
    manifest_path = os.path.join(HTML_OUTPUT_DIR, "download_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. Run part1_html_downloader.py first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    manifest_rows = load_manifest()
    logger.info("Loaded %d rows from manifest", len(manifest_rows))

    image_rows = asyncio.run(scrape_all(manifest_rows))
    logger.info("Scraped %d qualifying images (>= %dx%d)", len(image_rows), MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT)

    run_ocr_pipeline(image_rows)


if __name__ == "__main__":
    main()
