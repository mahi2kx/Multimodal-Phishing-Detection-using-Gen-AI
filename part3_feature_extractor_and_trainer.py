import os
import csv
import logging

import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

from config import (
    HTML_OUTPUT_DIR,
    OCR_OUTPUT_CSV,
    FEATURES_OUTPUT_XLSX,
    MODEL_OUTPUT_PATH,
    EMBEDDINGS_CACHE,
    ROBERTA_MODEL_NAME,
    MAX_TOKEN_LENGTH,
    EMBEDDING_BATCH_SIZE,
    TEST_SPLIT_SIZE,
    RANDOM_STATE,
    USE_GPU_IF_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Merge HTML text + OCR text per URL
# ---------------------------------------------------------------------------
def load_html_text(filename: str) -> str:
    if not filename:
        return ""
    path = os.path.join(HTML_OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def strip_html_tags(html: str) -> str:
    """Lightweight tag stripper (avoids adding a heavy BeautifulSoup dependency requirement)."""
    import re
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_combined_dataset() -> pd.DataFrame:
    manifest_path = os.path.join(HTML_OUTPUT_DIR, "download_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"{manifest_path} not found. Run part1 first.")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "success"].copy()

    # Aggregate OCR text per URL (a page can have multiple images)
    if os.path.exists(OCR_OUTPUT_CSV):
        ocr_df = pd.read_csv(OCR_OUTPUT_CSV)
        ocr_agg = (
            ocr_df.groupby("url")["ocr_text"]
            .apply(lambda texts: " ".join(str(t) for t in texts if pd.notna(t)))
            .reset_index()
        )
    else:
        logger.warning("%s not found — proceeding with HTML-only text. Run part2 for full multimodal features.", OCR_OUTPUT_CSV)
        ocr_agg = pd.DataFrame(columns=["url", "ocr_text"])

    merged = manifest.merge(ocr_agg, on="url", how="left")
    merged["ocr_text"] = merged["ocr_text"].fillna("")

    merged["html_raw"] = merged["filename"].apply(load_html_text)
    merged["html_text"] = merged["html_raw"].apply(strip_html_tags)

    merged["combined_text"] = (merged["html_text"] + " " + merged["ocr_text"]).str.strip()
    merged = merged[merged["combined_text"].str.len() > 0]

    return merged[["url", "label", "combined_text"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: RoBERTa embeddings
# ---------------------------------------------------------------------------
def get_device():
    if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_embeddings(texts: list[str]) -> np.ndarray:
    device = get_device()
    logger.info("Loading RoBERTa (%s) on %s", ROBERTA_MODEL_NAME, device)

    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = RobertaModel.from_pretrained(ROBERTA_MODEL_NAME).to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start:start + EMBEDDING_BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                return_tensors="pt",
            ).to(device)

            output = model(**encoded)
            # Mean-pool token embeddings (masking out padding) -> one vector per document
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            pooled = (summed / counts).cpu().numpy()

            all_embeddings.append(pooled)
            logger.info("Embedded %d/%d documents", min(start + EMBEDDING_BATCH_SIZE, len(texts)), len(texts))

    return np.vstack(all_embeddings)


def get_or_build_embeddings(df: pd.DataFrame) -> np.ndarray:
    if os.path.exists(EMBEDDINGS_CACHE):
        logger.info("Loading cached embeddings from %s", EMBEDDINGS_CACHE)
        cached = np.load(EMBEDDINGS_CACHE)
        if cached["embeddings"].shape[0] == len(df):
            return cached["embeddings"]
        logger.warning("Cache size mismatch with current dataset — regenerating embeddings.")

    embeddings = generate_embeddings(df["combined_text"].tolist())
    np.savez_compressed(EMBEDDINGS_CACHE, embeddings=embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# Step 3: Train classifier
# ---------------------------------------------------------------------------
def train_classifier(embeddings: np.ndarray, labels: pd.Series):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    logger.info("\n%s", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_proba)
        logger.info("ROC-AUC: %.4f", auc)
    except ValueError:
        pass  # only one class present in y_test (small dataset edge case)

    return clf, label_encoder


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("Building combined HTML + OCR dataset...")
    df = build_combined_dataset()
    logger.info("Dataset ready: %d rows", len(df))

    if len(df) < 10:
        logger.warning(
            "Very small dataset (%d rows). Results will not be meaningful until "
            "you run the full pipeline on your real URL list.", len(df)
        )

    logger.info("Generating/loading RoBERTa embeddings...")
    embeddings = get_or_build_embeddings(df)

    # Save features to Excel for inspection (embedding dims flattened to columns)
    feature_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    features_df = pd.concat(
        [df[["url", "label"]].reset_index(drop=True), pd.DataFrame(embeddings, columns=feature_cols)],
        axis=1,
    )
    features_df.to_excel(FEATURES_OUTPUT_XLSX, index=False)
    logger.info("Features saved to %s", FEATURES_OUTPUT_XLSX)

    logger.info("Training classifier...")
    clf, label_encoder = train_classifier(embeddings, df["label"])

    joblib.dump({"model": clf, "label_encoder": label_encoder}, MODEL_OUTPUT_PATH)
    logger.info("Model saved to %s", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
