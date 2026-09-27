import csv
import re
import unicodedata
from collections import Counter, defaultdict


S1_FILE = "dataset/extracted/student_resource/dataset/train/train_source1.tsv"
S2_FILE = "dataset/extracted/student_resource/dataset/train/train_source2.tsv"
S3_FILE = "dataset/extracted/student_resource/dataset/train/train_source3.tsv"
GT_FILE = "dataset/extracted/student_resource/dataset/train/train_ground_truth.tsv"

MIN_FREQ = 2
MAX_FREQ = 500


# ============================================================
# NORMALIZATION
# ============================================================

def norm_text(s):
    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def address_tokens(s):
    s = norm_text(s)

    if not s:
        return []

    return [
        token
        for token in s.split()
        if len(token) >= 2
    ]


def numeric_tokens(s):
    if not s:
        return []

    return re.findall(r"\d+", s)


# ============================================================
# LOAD TSV
# ============================================================

def load_tsv(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(path):

    gt = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        print(
            "Ground-truth columns:",
            reader.fieldnames
        )

        for row in reader:

            s1_id = row["source1_entity_id"]

            raw = row["matched_entity_ids"].strip()

            if raw:
                matches = {
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                }
            else:
                matches = set()

            gt[s1_id] = matches

    return gt


# ============================================================
# EXACT NAME INDEX
# ============================================================

def build_name_index(target):

    index = defaultdict(list)

    for i, row in enumerate(target):

        key = norm_text(
            row["business_name"]
        )

        if key:
            index[key].append(i)

    return index


# ============================================================
# ADDRESS TOKEN FREQUENCY
# ============================================================

def build_address_frequency(target):

    frequency = Counter()

    for row in target:

        tokens = set(
            address_tokens(
                row["business_address"]
            )
        )

        for token in tokens:
            frequency[token] += 1

    return frequency


# ============================================================
# ADDRESS INDEX
#
# country + numeric token + selective address token
# ============================================================

def build_address_index(
    target,
    frequency
):

    index = defaultdict(list)

    for i, row in enumerate(target):

        address = row["business_address"]

        if not address:
            continue

        nums = set(
            numeric_tokens(address)
        )

        tokens = set(
            address_tokens(address)
        )

        valid_tokens = [
            token
            for token in tokens
            if (
                MIN_FREQ
                <= frequency[token]
                <= MAX_FREQ
            )
        ]

        country = row["country"].casefold()

        for number in nums:

            for token in valid_tokens:

                key = (
                    country,
                    number,
                    token
                )

                index[key].append(i)

    return index


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, p):

    if not values:
        return 0

    values = sorted(values)

    k = (len(values) - 1) * p

    lo = int(k)
    hi = min(
        lo + 1,
        len(values) - 1
    )

    if lo == hi:
        return values[lo]

    return (
        values[lo]
        +
        (
            values[hi]
            - values[lo]
        )
        *
        (k - lo)
    )


# ============================================================
# BASELINE EVALUATION
# ============================================================

def evaluate(
    s1,
    target,
    gt,
    name_index,
    address_index,
    address_frequency,
    label
):

    print()
    print("=" * 70)
    print(f"VALIDATING BASELINE: {label}")
    print("=" * 70)

    total_candidates = 0
    total_true = 0
    recovered = 0

    candidate_counts = []

    zero_candidates = 0

    # Track contributions separately
    name_recovered = 0
    address_recovered = 0

    # Track union correctly
    union_recovered = 0

    for i, row in enumerate(s1):

        s1_id = row["entity_id"]

        true_ids = gt.get(
            s1_id,
            set()
        )

        total_true += len(true_ids)

        name_candidates = set()
        address_candidates = set()

        # ====================================================
        # EXACT NAME
        # ====================================================

        name_key = norm_text(
            row["business_name"]
        )

        for idx in name_index.get(
            name_key,
            []
        ):

            name_candidates.add(
                target[idx]["entity_id"]
            )

        # ====================================================
        # ADDRESS
        # ====================================================

        address = row["business_address"]

        if address:

            nums = set(
                numeric_tokens(address)
            )

            tokens = set(
                address_tokens(address)
            )

            valid_tokens = [
                token
                for token in tokens
                if (
                    MIN_FREQ
                    <= address_frequency[token]
                    <= MAX_FREQ
                )
            ]

            country = row["country"].casefold()

            for number in nums:

                for token in valid_tokens:

                    key = (
                        country,
                        number,
                        token
                    )

                    for idx in address_index.get(
                        key,
                        []
                    ):

                        address_candidates.add(
                            target[idx]["entity_id"]
                        )

        # ====================================================
        # UNION
        # ====================================================

        candidates = (
            name_candidates
            |
            address_candidates
        )

        candidate_count = len(candidates)

        candidate_counts.append(
            candidate_count
        )

        total_candidates += candidate_count

        if candidate_count == 0:
            zero_candidates += 1

        # ====================================================
        # RECALL
        # ====================================================

        name_hit = (
            true_ids
            &
            name_candidates
        )

        address_hit = (
            true_ids
            &
            address_candidates
        )

        union_hit = (
            true_ids
            &
            candidates
        )

        name_recovered += len(
            name_hit
        )

        address_recovered += len(
            address_hit
        )

        union_recovered += len(
            union_hit
        )

        if (i + 1) % 100000 == 0:

            print(
                f"Processed "
                f"{i + 1:,} / "
                f"{len(s1):,}",
                flush=True
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("-" * 70)

    print(
        f"S1 rows                 : "
        f"{len(s1):,}"
    )

    print(
        f"Target rows             : "
        f"{len(target):,}"
    )

    print(
        f"Total candidates        : "
        f"{total_candidates:,}"
    )

    print(
        f"Mean candidates/S1      : "
        f"{total_candidates / len(s1):.2f}"
    )

    print(
        f"P50 candidates/S1       : "
        f"{percentile(candidate_counts, 0.50):.0f}"
    )

    print(
        f"P90 candidates/S1       : "
        f"{percentile(candidate_counts, 0.90):.0f}"
    )

    print(
        f"P95 candidates/S1       : "
        f"{percentile(candidate_counts, 0.95):.0f}"
    )

    print(
        f"P99 candidates/S1       : "
        f"{percentile(candidate_counts, 0.99):.0f}"
    )

    print(
        f"P99.9 candidates/S1     : "
        f"{percentile(candidate_counts, 0.999):.0f}"
    )

    print(
        f"MAX candidates/S1       : "
        f"{max(candidate_counts):,}"
    )

    print(
        f"Zero-candidate S1       : "
        f"{zero_candidates:,}"
    )

    print()

    print(
        f"Total true matches      : "
        f"{total_true:,}"
    )

    print(
        f"Name recovered          : "
        f"{name_recovered:,} "
        f"("
        f"{name_recovered / total_true * 100:.4f}%"
        f")"
    )

    print(
        f"Address recovered       : "
        f"{address_recovered:,} "
        f"("
        f"{address_recovered / total_true * 100:.4f}%"
        f")"
    )

    print(
        f"UNION recovered         : "
        f"{union_recovered:,} "
        f"("
        f"{union_recovered / total_true * 100:.4f}%"
        f")"
    )

    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BASELINE BLOCKING VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    print()
    print("Loading S1...")

    s1 = load_tsv(S1_FILE)

    print(
        f"S1: {len(s1):,}"
    )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    print()
    print("Loading ground truth...")

    gt = load_ground_truth(
        GT_FILE
    )

    print(
        f"GT: {len(gt):,}"
    )

    # ========================================================
    # S2
    # ========================================================

    print()
    print("#" * 70)
    print("S2")
    print("#" * 70)

    print()
    print("Loading S2...")

    s2 = load_tsv(S2_FILE)

    print(
        f"S2: {len(s2):,}"
    )

    print()
    print("Building S2 name index...")

    name_index = build_name_index(
        s2
    )

    print(
        f"Name keys: "
        f"{len(name_index):,}"
    )

    print()
    print("Building S2 address frequencies...")

    address_frequency = build_address_frequency(
        s2
    )

    print(
        f"Address tokens: "
        f"{len(address_frequency):,}"
    )

    print()
    print("Building S2 address index...")

    address_index = build_address_index(
        s2,
        address_frequency
    )

    print(
        f"Address keys: "
        f"{len(address_index):,}"
    )

    print()
    print("Evaluating S2...")

    evaluate(
        s1=s1,
        target=s2,
        gt=gt,
        name_index=name_index,
        address_index=address_index,
        address_frequency=address_frequency,
        label="S2 | Name + Address 2-500"
    )

    # --------------------------------------------------------
    # Release S2
    # --------------------------------------------------------

    del s2
    del name_index
    del address_index
    del address_frequency

    # ========================================================
    # S3
    # ========================================================

    print()
    print("#" * 70)
    print("S3")
    print("#" * 70)

    print()
    print("Loading S3...")

    s3 = load_tsv(S3_FILE)

    print(
        f"S3: {len(s3):,}"
    )

    print()
    print("Building S3 name index...")

    name_index = build_name_index(
        s3
    )

    print(
        f"Name keys: "
        f"{len(name_index):,}"
    )

    print()
    print("Building S3 address frequencies...")

    address_frequency = build_address_frequency(
        s3
    )

    print(
        f"Address tokens: "
        f"{len(address_frequency):,}"
    )

    print()
    print("Building S3 address index...")

    address_index = build_address_index(
        s3,
        address_frequency
    )

    print(
        f"Address keys: "
        f"{len(address_index):,}"
    )

    print()
    print("Evaluating S3...")

    evaluate(
        s1=s1,
        target=s3,
        gt=gt,
        name_index=name_index,
        address_index=address_index,
        address_frequency=address_frequency,
        label="S3 | Name + Address 2-500"
    )


if __name__ == "__main__":
    main()
