import csv
import re
import unicodedata
from collections import Counter, defaultdict


S1_FILE = "dataset/extracted/student_resource/dataset/train/train_source1.tsv"
S2_FILE = "dataset/extracted/student_resource/dataset/train/train_source2.tsv"
S3_FILE = "dataset/extracted/student_resource/dataset/train/train_source3.tsv"
GT_FILE = "dataset/extracted/student_resource/dataset/train/train_ground_truth.tsv"


# ============================================================
# CONFIGURATION
# ============================================================

NAME_MIN_FREQ = 2
NAME_MAX_FREQ = 50

ADDR_MIN_FREQ = 2
ADDR_MAX_FREQ = 500


NAME_STOPWORDS = {
    "the", "and", "of", "at", "in", "for", "to",
    "llc", "inc", "corp", "corporation", "company", "co",
    "limited", "ltd", "plc", "llp", "pvt", "private",
    "group", "services", "service", "center", "centre",
    "medical", "health", "healthcare",
    "hospital", "clinic", "care",
    "dental", "pharmacy",
    "therapy", "therapies",
    "practice", "practices",
    "general", "primary", "specialty", "specialist",
    "professional", "professionals",
}


# ============================================================
# NORMALIZATION
# ============================================================

def norm_text(s):
    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()

    # Preserve Unicode letters/numbers.
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)

    s = re.sub(r"\s+", " ", s).strip()

    return s


def name_tokens(s):
    s = norm_text(s)

    if not s:
        return []

    return [
        token
        for token in s.split()
        if len(token) >= 2 and token not in NAME_STOPWORDS
    ]


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
# TSV LOADING
# ============================================================

def load_tsv(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(path):
    """
    Automatically detects the ground-truth columns.

    Expected structure is conceptually:

        S1_ID    S2_IDs / matches

    where the second column contains comma-separated
    matching target IDs.
    """

    gt = {}

    with open(path, "r", encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        fieldnames = reader.fieldnames

        print()
        print("Ground-truth columns:")
        print(fieldnames)

        if not fieldnames:
            raise RuntimeError("Ground-truth file has no header.")

        # ----------------------------------------------------
        # Detect S1 ID column
        # ----------------------------------------------------

        id_candidates = [
            "id",
            "entity_id",
            "source1_id",
            "s1_id",
            "business_id",
        ]

        id_column = None

        for candidate in id_candidates:
            if candidate in fieldnames:
                id_column = candidate
                break

        # If standard names fail, use first column.
        if id_column is None:
            id_column = fieldnames[0]

        # ----------------------------------------------------
        # Detect matches column
        # ----------------------------------------------------

        match_candidates = [
            "matches",
            "match_ids",
            "matching_ids",
            "s2_ids",
            "s3_ids",
            "source2_ids",
            "source3_ids",
        ]

        match_column = None

        for candidate in match_candidates:
            if candidate in fieldnames:
                match_column = candidate
                break

        # If standard names fail, use second column.
        if match_column is None:

            if len(fieldnames) >= 2:
                match_column = fieldnames[1]
            else:
                raise RuntimeError(
                    "Could not determine ground-truth match column."
                )

        print(f"Using S1 ID column : {id_column}")
        print(f"Using match column  : {match_column}")

        # ----------------------------------------------------
        # Read GT
        # ----------------------------------------------------

        for row in reader:

            s1_id = row[id_column].strip()

            raw_matches = row[match_column].strip()

            if raw_matches:
                matches = {
                    x.strip()
                    for x in raw_matches.split(",")
                    if x.strip()
                }
            else:
                matches = set()

            gt[s1_id] = matches

    return gt


# ============================================================
# NAME INDEX
# ============================================================

def build_name_index(rows):

    index = defaultdict(list)

    for i, row in enumerate(rows):

        key = norm_text(
            row["business_name"]
        )

        if key:
            index[key].append(i)

    return index


# ============================================================
# NAME TOKEN FREQUENCY
# ============================================================

def build_name_token_frequency(rows):

    counter = Counter()

    for row in rows:

        tokens = set(
            name_tokens(
                row["business_name"]
            )
        )

        for token in tokens:
            counter[token] += 1

    return counter


# ============================================================
# NAME TOKEN INDEX
# ============================================================

def build_name_token_index(rows, frequency):

    index = defaultdict(list)

    for i, row in enumerate(rows):

        tokens = set(
            name_tokens(
                row["business_name"]
            )
        )

        for token in tokens:

            freq = frequency[token]

            if (
                NAME_MIN_FREQ
                <= freq
                <= NAME_MAX_FREQ
            ):
                index[token].append(i)

    return index


# ============================================================
# ADDRESS TOKEN FREQUENCY
# ============================================================

def build_address_token_frequency(rows):

    counter = Counter()

    for row in rows:

        tokens = set(
            address_tokens(
                row["business_address"]
            )
        )

        for token in tokens:
            counter[token] += 1

    return counter


# ============================================================
# ADDRESS INDEX
# ============================================================

def build_address_index(rows, frequency):

    index = defaultdict(list)

    for i, row in enumerate(rows):

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
                ADDR_MIN_FREQ
                <= frequency[token]
                <= ADDR_MAX_FREQ
            )
        ]

        country = row["country"].casefold()

        for num in nums:

            for token in valid_tokens:

                key = (
                    country,
                    num,
                    token,
                )

                index[key].append(i)

    return index


# ============================================================
# PERCENTILES
# ============================================================

def percentile(values, p):

    if not values:
        return 0

    values = sorted(values)

    k = (len(values) - 1) * p

    lower = int(k)
    upper = min(
        lower + 1,
        len(values) - 1
    )

    if lower == upper:
        return values[lower]

    return (
        values[lower]
        +
        (
            values[upper]
            - values[lower]
        )
        *
        (k - lower)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    s1,
    target,
    gt,
    name_index,
    addr_index,
    token_index,
    name_frequency,
    addr_frequency,
):

    candidate_counts = []

    total_candidates = 0

    total_true = 0

    recovered_name = 0
    recovered_addr = 0
    recovered_token = 0
    recovered_union = 0

    incremental_name = 0
    incremental_addr = 0
    incremental_token = 0

    zero_candidates = 0

    for i, row in enumerate(s1):

        s1_id = row["entity_id"]

        true_ids = gt.get(
            s1_id,
            set()
        )

        total_true += len(true_ids)

        # ----------------------------------------------------
        # Candidate sets
        # ----------------------------------------------------

        name_candidates = set()
        addr_candidates = set()
        token_candidates = set()

        # ====================================================
        # 1. EXACT NORMALIZED NAME
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
        # 2. ADDRESS ANCHOR
        #
        # country + numeric token + address token
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
                    ADDR_MIN_FREQ
                    <= addr_frequency[token]
                    <= ADDR_MAX_FREQ
                )
            ]

            country = row["country"].casefold()

            for num in nums:

                for token in valid_tokens:

                    key = (
                        country,
                        num,
                        token,
                    )

                    for idx in addr_index.get(
                        key,
                        []
                    ):

                        addr_candidates.add(
                            target[idx]["entity_id"]
                        )

        # ====================================================
        # 3. SELECTIVE NAME TOKEN
        # ====================================================

        for token in set(
            name_tokens(
                row["business_name"]
            )
        ):

            freq = name_frequency[token]

            if (
                NAME_MIN_FREQ
                <= freq
                <= NAME_MAX_FREQ
            ):

                for idx in token_index.get(
                    token,
                    []
                ):

                    token_candidates.add(
                        target[idx]["entity_id"]
                    )

        # ====================================================
        # UNION
        # ====================================================

        candidates = (
            name_candidates
            |
            addr_candidates
            |
            token_candidates
        )

        candidate_counts.append(
            len(candidates)
        )

        total_candidates += len(candidates)

        if not candidates:
            zero_candidates += 1

        # ====================================================
        # RECALL
        # ====================================================

        name_hit = (
            true_ids
            &
            name_candidates
        )

        addr_hit = (
            true_ids
            &
            addr_candidates
        )

        token_hit = (
            true_ids
            &
            token_candidates
        )

        union_hit = (
            true_ids
            &
            candidates
        )

        recovered_name += len(
            name_hit
        )

        recovered_addr += len(
            addr_hit
        )

        recovered_token += len(
            token_hit
        )

        recovered_union += len(
            union_hit
        )

        # ====================================================
        # INCREMENTAL CONTRIBUTION
        # ====================================================

        name_addr = (
            name_candidates
            |
            addr_candidates
        )

        incremental_token += len(
            true_ids
            &
            token_candidates
            -
            name_addr
        )

        incremental_addr += len(
            true_ids
            &
            addr_candidates
            -
            name_candidates
        )

        incremental_name += len(
            true_ids
            &
            name_candidates
            -
            addr_candidates
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

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
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

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
        f"{recovered_name:,} "
        f"("
        f"{recovered_name / total_true * 100:.4f}%"
        f")"
    )

    print(
        f"Address recovered       : "
        f"{recovered_addr:,} "
        f"("
        f"{recovered_addr / total_true * 100:.4f}%"
        f")"
    )

    print(
        f"Name-token recovered    : "
        f"{recovered_token:,} "
        f"("
        f"{recovered_token / total_true * 100:.4f}%"
        f")"
    )

    print(
        f"UNION recovered         : "
        f"{recovered_union:,} "
        f"("
        f"{recovered_union / total_true * 100:.4f}%"
        f")"
    )

    print()

    print(
        f"Incremental name        : "
        f"{incremental_name:,}"
    )

    print(
        f"Incremental address     : "
        f"{incremental_addr:,}"
    )

    print(
        f"Incremental name-token  : "
        f"{incremental_token:,}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UNION BLOCKING EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    print()
    print("Loading S1...")

    s1 = load_tsv(
        S1_FILE
    )

    print(
        f"S1 loaded: {len(s1):,}"
    )

    # --------------------------------------------------------
    # Load S2
    # --------------------------------------------------------

    print()
    print("Loading target S2...")

    s2 = load_tsv(
        S2_FILE
    )

    print(
        f"S2 loaded: {len(s2):,}"
    )

    # --------------------------------------------------------
    # Load S3
    # --------------------------------------------------------

    print()
    print("Loading target S3...")

    s3 = load_tsv(
        S3_FILE
    )

    print(
        f"S3 loaded: {len(s3):,}"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print()
    print("Loading ground truth...")

    gt = load_ground_truth(
        GT_FILE
    )

    print(
        f"Ground truth rows: "
        f"{len(gt):,}"
    )

    # ========================================================
    # S2
    # ========================================================

    print()
    print("#" * 70)
    print("S2 EXPERIMENT")
    print("#" * 70)

    print()
    print("Building S2 exact-name index...")

    name_index_s2 = build_name_index(
        s2
    )

    print(
        f"S2 name keys: "
        f"{len(name_index_s2):,}"
    )

    print()
    print("Building S2 name-token frequencies...")

    name_freq_s2 = build_name_token_frequency(
        s2
    )

    print(
        f"S2 name tokens: "
        f"{len(name_freq_s2):,}"
    )

    print()
    print("Building S2 name-token index...")

    token_index_s2 = build_name_token_index(
        s2,
        name_freq_s2
    )

    print(
        f"S2 selective name tokens: "
        f"{len(token_index_s2):,}"
    )

    print()
    print("Building S2 address-token frequencies...")

    addr_freq_s2 = build_address_token_frequency(
        s2
    )

    print(
        f"S2 address tokens: "
        f"{len(addr_freq_s2):,}"
    )

    print()
    print("Building S2 address index...")

    addr_index_s2 = build_address_index(
        s2,
        addr_freq_s2
    )

    print(
        f"S2 address keys: "
        f"{len(addr_index_s2):,}"
    )

    print()
    print("Evaluating S2...")

    evaluate(
        s1=s1,
        target=s2,
        gt=gt,
        name_index=name_index_s2,
        addr_index=addr_index_s2,
        token_index=token_index_s2,
        name_frequency=name_freq_s2,
        addr_frequency=addr_freq_s2,
    )

    # ========================================================
    # Free S2 memory before S3
    # ========================================================

    print()
    print("Releasing S2 indexes...")

    del name_index_s2
    del token_index_s2
    del addr_index_s2
    del name_freq_s2
    del addr_freq_s2
    del s2

    # ========================================================
    # S3
    # ========================================================

    print()
    print("#" * 70)
    print("S3 EXPERIMENT")
    print("#" * 70)

    print()
    print("Building S3 exact-name index...")

    name_index_s3 = build_name_index(
        s3
    )

    print(
        f"S3 name keys: "
        f"{len(name_index_s3):,}"
    )

    print()
    print("Building S3 name-token frequencies...")

    name_freq_s3 = build_name_token_frequency(
        s3
    )

    print(
        f"S3 name tokens: "
        f"{len(name_freq_s3):,}"
    )

    print()
    print("Building S3 name-token index...")

    token_index_s3 = build_name_token_index(
        s3,
        name_freq_s3
    )

    print(
        f"S3 selective name tokens: "
        f"{len(token_index_s3):,}"
    )

    print()
    print("Building S3 address-token frequencies...")

    addr_freq_s3 = build_address_token_frequency(
        s3
    )

    print(
        f"S3 address tokens: "
        f"{len(addr_freq_s3):,}"
    )

    print()
    print("Building S3 address index...")

    addr_index_s3 = build_address_index(
        s3,
        addr_freq_s3
    )

    print(
        f"S3 address keys: "
        f"{len(addr_index_s3):,}"
    )

    print()
    print("Evaluating S3...")

    evaluate(
        s1=s1,
        target=s3,
        gt=gt,
        name_index=name_index_s3,
        addr_index=addr_index_s3,
        token_index=token_index_s3,
        name_frequency=name_freq_s3,
        addr_frequency=addr_freq_s3,
    )


if __name__ == "__main__":
    main()
