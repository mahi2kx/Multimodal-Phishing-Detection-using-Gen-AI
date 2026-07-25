"""
predict.py
-----------
Run the trained model against a single new URL end-to-end:
downloads HTML, scrapes+OCRs images, embeds combined text, predicts label.

Usage:
    python predict.py https://example-site.com
"""

import sys
import asyncio
import logging

import numpy as np
import joblib

from config import MODEL_OUTPUT_PATH
from part1_html_downloader import download_html
from part2_image_scraper import get_ocr_reader
from part3_feature_extractor_and_trainer import strip_html_tags, generate_embeddings

from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def get_page_ocr_text(url: str) -> str:
    """Lightweight single-URL version of the part2 scraping logic."""
    reader = get_ocr_reader()
    texts = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1366, "height": 900})
        try:
            await page.goto(url, timeout=20000, wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)
            imgs = await page.query_selector_all("img")
            for i, img in enumerate(imgs):
                box = await img.bounding_box()
                if not box or box["width"] < 200 or box["height"] < 200:
                    continue
                tmp_path = f"/tmp/predict_img_{i}.png"
                await img.screenshot(path=tmp_path)
                result = reader.readtext(tmp_path, detail=0)
                texts.append(" ".join(result))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image/OCR step failed: %s", exc)
        finally:
            await browser.close()
    return " ".join(texts)


def predict(url: str):
    logger.info("Downloading HTML for %s", url)
    success, html_or_error = download_html(url)
    html_text = strip_html_tags(html_or_error) if success else ""
    if not success:
        logger.warning("HTML download failed: %s", html_or_error)

    logger.info("Scraping images + running OCR...")
    ocr_text = asyncio.run(get_page_ocr_text(url))

    combined_text = (html_text + " " + ocr_text).strip()
    if not combined_text:
        logger.error("No text could be extracted from this URL. Cannot predict.")
        return

    logger.info("Generating embedding...")
    embedding = generate_embeddings([combined_text])

    logger.info("Loading model from %s", MODEL_OUTPUT_PATH)
    saved = joblib.load(MODEL_OUTPUT_PATH)
    clf, label_encoder = saved["model"], saved["label_encoder"]

    pred_idx = clf.predict(embedding)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    proba = clf.predict_proba(embedding)[0]

    print("\n=== Prediction Result ===")
    print(f"URL:        {url}")
    print(f"Prediction: {pred_label}")
    for cls, p in zip(label_encoder.classes_, proba):
        print(f"  {cls}: {p:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <url>")
        sys.exit(1)
    predict(sys.argv[1])
