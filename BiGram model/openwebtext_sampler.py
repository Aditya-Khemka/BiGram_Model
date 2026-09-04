import os
import tarfile
import lzma
import random
from tqdm import tqdm
import concurrent.futures

# ---------------- CONFIG ----------------
BASE_DIR = r"C:\Users\USER\Desktop\ML\openwebtext\subsets"
TRAIN_OUT = "output_train.txt"
VAL_OUT   = "output_val.txt"
VOCAB_OUT = "vocab.txt"

TRAIN_RATIO = 0.9
SAMPLE_RATE = None     # set to e.g. 0.01 to sample docs, or None to disable
MAX_WORKERS = 8
RANDOM_SEED = 42
# ----------------------------------------

random.seed(RANDOM_SEED)


# ========== DISCOVERY ==========
def list_subset_tars(base_dir):
    print("[MAIN] Scanning subsets directory")
    tars = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith(".tar")
    ]
    print(f"[MAIN] Total tar files found: {len(tars)}")
    return sorted(tars)


def list_all_documents(tar_paths):
    """
    Returns list of (tar_path, xz_member_name)
    """
    documents = []

    for tar_path in tar_paths:
        print(f"[TAR] Opening {os.path.basename(tar_path)}")
        with tarfile.open(tar_path, "r") as tar:
            xz_members = [
                m.name for m in tar.getmembers()
                if m.name.endswith(".xz")
            ]
        print(f"[TAR] Found {len(xz_members)} .xz files")
        for xz in xz_members:
            documents.append((tar_path, xz))
        print(f"[TAR] Finished {os.path.basename(tar_path)}\n")

    print(f"[MAIN] Total documents found: {len(documents)}")
    return documents


# ========== PROCESSING ==========
def process_document(args):
    tar_path, xz_member = args

    try:
        with tarfile.open(tar_path, "r") as tar:
            member = tar.getmember(xz_member)
            f = tar.extractfile(member)
            if f is None:
                return "", set()

            with lzma.open(f, "rt", encoding="utf-8", errors="ignore") as infile:
                text = infile.read()

        return text, set(text)

    except Exception:
        # Never kill the executor
        return "", set()


def process_split(docs, output_path, label):
    vocab = set()

    print(f"[MAIN] Starting {label} processing")
    with open(output_path, "a", encoding="utf-8") as outfile:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            for text, chars in tqdm(
                executor.map(process_document, docs),
                total=len(docs),
                desc=f"{label}"
            ):
                outfile.write(text)
                vocab.update(chars)

    return vocab


# ========== MAIN ==========
if __name__ == "__main__":

    # Clear outputs
    open(TRAIN_OUT, "w").close()
    open(VAL_OUT, "w").close()
    print("[MAIN] Output files cleared\n")

    # Discover data
    tar_files = list_subset_tars(BASE_DIR)
    documents = list_all_documents(tar_files)

    # Shuffle documents (critical for faithful replication)
    random.shuffle(documents)

    # Train / Val split at DOCUMENT level
    split_idx = int(len(documents) * TRAIN_RATIO)
    train_docs = documents[:split_idx]
    val_docs   = documents[split_idx:]

    print(f"[MAIN] Train documents: {len(train_docs)}")
    print(f"[MAIN] Val documents:   {len(val_docs)}")

    # Optional sampling (document-level, same as original)
    if SAMPLE_RATE is not None:
        def sample(docs):
            return random.sample(
                docs,
                max(1, int(len(docs) * SAMPLE_RATE))
            )

        train_docs = sample(train_docs)
        val_docs   = sample(val_docs)

        print(f"[MAIN] Sampled train docs: {len(train_docs)}")
        print(f"[MAIN] Sampled val docs:   {len(val_docs)}")

    print()

    # Process
    vocab_train = process_split(train_docs, TRAIN_OUT, "TRAIN")
    vocab_val   = process_split(val_docs,   VAL_OUT,   "VAL")

    # Write vocab
    vocab = vocab_train.union(vocab_val)
    print(f"\n[MAIN] Writing vocab with {len(vocab)} unique characters")

    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        for c in sorted(vocab):
            f.write(c + "\n")

    print("[MAIN] DONE")
