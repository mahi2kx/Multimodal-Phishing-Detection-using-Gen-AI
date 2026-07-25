# Multimodal-Phishing-Detection-using-Gen-AI
Detect phishing websites by combining **HTML text**, **visual content**, and **OCR text extracted from images** — because modern phishing pages hide malicious intent behind deceptive UI, screenshots, and text baked into images that plain HTML scrapers miss.

---

## 🚀 Project Overview

Online scams increasingly rely on fake websites that look almost indistinguishable from the real thing, stealing passwords, OTPs, and banking details. Traditional phishing detectors that rely only on HTML/text miss attacks where the deceptive content lives inside an image.

This project builds a **multimodal phishing detection system** that fuses:

- HTML content
- Images extracted from the webpage
- OCR text pulled from those images (EasyOCR)
- Deep contextual text embeddings (RoBERTa)
- A classifier trained on the combined feature set

The result: a system that catches phishing pages even when the malicious signal is hidden in a screenshot rather than the page's raw text.

---

## 🧠 Key Features

### ✅ 1. HTML Downloader — `part1_html_downloader.py`
- Automatically downloads raw HTML from each URL
- Saves content into text files
- Feeds the text-based feature extraction pipeline

### ✅ 2. Image Scraper — `part2_image_scraper.py`
- Uses **Playwright** to render and open webpages
- Extracts all images larger than 200×200px
- Runs OCR on each image to surface embedded text — critical for catching image-based phishing

### ✅ 3. OCR + Embedding + Training — `part3_feature_extractor_and_trainer.py`
- Reads HTML text + OCR text together
- Converts combined text into embeddings using **RoBERTa**
- Trains a classifier to distinguish phishing vs. legitimate URLs
- Saves the embeddings and the trained model

---

## 📂 Folder Structure

```
multimodal-phishing-detection/
│
├── config.py                     # Central configuration (paths, hyperparams)
├── part1_html_downloader.py
├── part2_image_scraper.py
├── part3_feature_extractor_and_trainer.py
├── predict.py                    # Run the trained model on a single new URL
│
├── dataset/
│   └── urls.csv                  # Input: url,label
│
├── Alloutputs/                   # Downloaded HTML files + manifest
├── images/                       # Extracted images
├── ocr_output.csv                # OCR text per image
├── features_output.xlsx          # Combined feature set (embeddings)
├── embeddings_cache.npz          # Cached RoBERTa embeddings
├── phishing_clf.joblib           # Trained model (after training)
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Category            | Tools/Libraries                          |
|---------------------|-------------------------------------------|
| Language             | Python                                    |
| Web Automation       | Playwright                                |
| OCR                  | EasyOCR                                   |
| Data Handling        | Pandas                                    |
| Deep Learning        | PyTorch, RoBERTa                          |
| ML / Classification  | Scikit-learn                              |
| Concurrency          | Multithreading, Asyncio                   |

---

## 📊 Dataset

- **34,000+ URLs** sourced from:
  - Mendeley Phishing Dataset
  - Various trusted legitimate sites
- Each entry includes:
  - HTML content
  - Images (converted to text via OCR)
  - Label: `phishing` / `legitimate`

---

## 🔧 How to Run

**Step 0 — Add your URLs**
Populate `dataset/urls.csv` with columns `url,label` (label = `phishing` or `legitimate`).

**Step 1 — Download HTML**
```bash
python part1_html_downloader.py
```

**Step 2 — Extract Images**
```bash
python part2_image_scraper.py
```

**Step 3 — Train the Model**
```bash
python part3_feature_extractor_and_trainer.py
```

**Step 4 — Predict on a New URL**
```bash
python predict.py https://example-site.com
```

---

## 🎯 Project Goal

Build a robust, scalable phishing detection system that analyzes:

- Text content
- Visual content
- OCR-extracted text
- Deep contextual meaning

**Applications:** Banking, E-commerce, Cybersecurity tooling

---

## 📌 Notes

- Ensure `requirements.txt` dependencies are installed before running the pipeline (`pip install -r requirements.txt`).
- Playwright requires an additional browser install step: `playwright install`.
- GPU is recommended for RoBERTa embedding generation at scale (34K+ URLs).
  
  Author
  BANOTH MAHESH
