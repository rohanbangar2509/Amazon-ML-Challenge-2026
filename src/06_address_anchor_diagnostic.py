import csv
import re
import unicodedata
from collections import Counter


S1_FILE = "dataset/extracted/student_resource/dataset/train/train_source1.tsv"
S2_FILE = "dataset/extracted/student_resource/dataset/train/train_source2.tsv"
S3_FILE = "dataset/extracted/student_resource/dataset/train/train_source3.tsv"


MIN_FREQ = 2
MAX_FREQ = 500


def norm_text(s):
    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def tokens(s):
    s = norm_text(s)

    if not s:
        return []

    return [
        x for x in s.split()
        if len(x) >= 2
    ]


def numbers(s):
    if not s:
        return []

    return re.findall(r"\d+", s)


def analyze(path, label):

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    frequency = Counter()

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

            address = row["business_address"]

            rows.append(address)

            seen = set(tokens(address))

            for token in seen:
                frequency[token] += 1

    print(
        f"Rows: {len(rows):,}"
    )

    # --------------------------------------------------------
    # Valid selective tokens
    # --------------------------------------------------------

    valid = {
        token
        for token, freq in frequency.items()
        if MIN_FREQ <= freq <= MAX_FREQ
    }

    print(
        f"All address tokens: "
        f"{len(frequency):,}"
    )

    print(
        f"Selective tokens "
        f"(2-500): {len(valid):,}"
    )

    # --------------------------------------------------------
    # Address anchor statistics
    # --------------------------------------------------------

    total_all = 0
    total_first = 0

    max_all = 0
    max_first = 0

    no_number = 0

    for address in rows:

        nums = numbers(address)

        addr_tokens = set(tokens(address))

        valid_tokens = [
            token
            for token in addr_tokens
            if token in valid
        ]

        # All numeric tokens
        all_combinations = (
            len(nums)
            *
            len(valid_tokens)
        )

        # First numeric token only
        first_combinations = (
            len(valid_tokens)
            if nums
            else 0
        )

        total_all += all_combinations
        total_first += first_combinations

        max_all = max(
            max_all,
            all_combinations
        )

        max_first = max(
            max_first,
            first_combinations
        )

        if not nums:
            no_number += 1

    print()
    print(
        "Address anchor key combinations:"
    )

    print(
        f"All numbers × tokens : "
        f"{total_all:,}"
    )

    print(
        f"First number × tokens: "
        f"{total_first:,}"
    )

    print(
        f"Ratio                 : "
        f"{total_all / max(total_first, 1):.2f}x"
    )

    print()
    print(
        f"Max all-number combinations  : "
        f"{max_all:,}"
    )

    print(
        f"Max first-number combinations: "
        f"{max_first:,}"
    )

    print(
        f"Addresses without number     : "
        f"{no_number:,}"
    )

    print()

    # --------------------------------------------------------
    # Sample addresses
    # --------------------------------------------------------

    print("Sample addresses:")
    print("-" * 70)

    shown = 0

    for address in rows:

        nums = numbers(address)

        if len(nums) >= 2:

            print(
                f"ADDRESS: {address}"
            )

            print(
                f"NUMBERS : {nums}"
            )

            print(
                f"TOKENS  : "
                f"{[x for x in tokens(address) if x in valid]}"
            )

            print()

            shown += 1

            if shown >= 10:
                break


def main():

    analyze(
        S2_FILE,
        "S2 ADDRESS DIAGNOSTIC"
    )

    analyze(
        S3_FILE,
        "S3 ADDRESS DIAGNOSTIC"
    )


if __name__ == "__main__":
    main()
