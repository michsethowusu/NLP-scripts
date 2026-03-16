"""
Speaker diarisation — outputs a CSV with speaker_id per audio file.

How it works:
  1. Loads each WAV and extracts a speaker embedding (256-dim voice fingerprint)
     using Resemblyzer's pretrained GE2E model.
  2. Clusters all embeddings with KMeans to identify N speakers.
  3. Writes a CSV: filename, speaker_id, duration_s

Install deps:
    pip install resemblyzer scikit-learn numpy
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these values, then run:  python diarise_wavs.py
# ══════════════════════════════════════════════════════════════════════════════

INPUT_DIR   = "8sec_voice_only_chunks"   # folder containing your WAV files
OUTPUT_CSV  = "diarisation.csv"          # path for the output CSV file

# Set to an integer to force a specific number of speakers (e.g. 3).
# Set to None to auto-detect (tests k=2–8 and picks the best silhouette score).
NUM_SPEAKERS = None

# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import csv
import wave
import warnings
import numpy as np

warnings.filterwarnings("ignore")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_encoder():
    from resemblyzer import VoiceEncoder
    print("Loading voice encoder (downloads ~17 MB on first run)…")
    return VoiceEncoder()


def embed_file(encoder, path):
    """Return a 256-dim L2-normalised speaker embedding for a WAV file."""
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(path)
    if len(wav) < 1600:   # skip files shorter than 0.1 s
        return None
    return encoder.embed_utterance(wav)


def get_duration(path):
    """Return duration in seconds for a WAV file."""
    try:
        with wave.open(path, "rb") as w:
            return round(w.getnframes() / w.getframerate(), 2)
    except Exception:
        return None


def best_n_clusters(embeddings, min_k=2, max_k=8):
    """Pick k via silhouette score if number of speakers is unknown."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    max_k = min(max_k, n - 1)
    if max_k < min_k:
        return min_k

    best_k, best_score = min_k, -1
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"  k={k}  silhouette={score:.3f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"-> Auto-selected {best_k} speaker(s)  (silhouette={best_score:.3f})\n")
    return best_k


def cluster(embeddings, n_speakers):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    wav_files = sorted(
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")
    )
    if not wav_files:
        print(f"No WAV files found in: {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(wav_files)} WAV file(s) in '{INPUT_DIR}'\n")

    # ── embed ──────────────────────────────────────────────────────────────
    encoder = load_encoder()
    embeddings, valid_files, durations = [], [], []

    for fname in wav_files:
        fpath = os.path.join(INPUT_DIR, fname)
        print(f"  Embedding: {fname}")
        emb = embed_file(encoder, fpath)
        if emb is None:
            print(f"    WARNING: Skipped (too short)")
            continue
        embeddings.append(emb)
        valid_files.append(fname)
        durations.append(get_duration(fpath))

    if not embeddings:
        print("No usable files after embedding.")
        sys.exit(1)

    embeddings = np.array(embeddings)

    # ── cluster ────────────────────────────────────────────────────────────
    print()
    if NUM_SPEAKERS:
        n = NUM_SPEAKERS
        print(f"Using {n} speaker(s) as specified.\n")
    else:
        print("Auto-detecting number of speakers...")
        n = best_n_clusters(embeddings)

    labels = cluster(embeddings, n)

    # ── write CSV ──────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "speaker_id", "duration_s"])
        for fname, label, dur in zip(valid_files, labels, durations):
            writer.writerow([fname, f"speaker_{label:02d}", dur])

    # ── summary ────────────────────────────────────────────────────────────
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    print(f"{'-'*50}")
    print(f"CSV written to: {OUTPUT_CSV}")
    print(f"{len(valid_files)} file(s) across {n} speaker(s):\n")
    for label in sorted(counts):
        print(f"  speaker_{label:02d}  ->  {counts[label]} file(s)")


if __name__ == "__main__":
    main()
