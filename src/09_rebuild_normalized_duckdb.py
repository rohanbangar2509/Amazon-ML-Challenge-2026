import os
import re
import csv
import unicodedata
import duckdb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.expanduser(
    "~/amazon_ml_challenge"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "extracted",
    "student_resource",
    "dataset"
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

TEST_DIR = os.path.join(
    DATA_DIR,
    "test"
)

WORKING_DIR = os.path.join(
    BASE_DIR,
    "working"
)

TEMP_DIR = os.path.join(
    BASE_DIR,
    "temp",
    "duckdb"
)

OUTPUT_DB = os.path.join(
    WORKING_DIR,
    "amazon_er_v2.duckdb"
)


# ============================================================
# SETTINGS
# ============================================================

MEMORY_LIMIT = "5GB"
THREADS = 6

CHUNK_SIZE = 100_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value)

    if not value:
        return ""

    # Unicode compatibility normalization.
    value = unicodedata.normalize(
        "NFKC",
        value
    )

    # Case-insensitive normalization.
    value = value.casefold()

    # Keep Unicode word characters and whitespace.
    #
    # This is intentionally NOT ASCII-only.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Collapse whitespace.
    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def normalize_country(value):

    if value is None:
        return ""

    return str(value).strip().casefold()


# ============================================================
# PROCESS ONE TSV
# ============================================================

def process_tsv(
    con,
    input_file,
    table_name
):

    print()
    print("=" * 70)
    print(
        f"PROCESSING: {table_name}"
    )
    print("=" * 70)

    print(
        f"Input: {input_file}"
    )

    # --------------------------------------------------------
    # Create destination table.
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} (

            entity_id VARCHAR,

            business_name VARCHAR,
            business_address VARCHAR,
            country VARCHAR,

            name_norm VARCHAR,
            address_norm VARCHAR,
            country_norm VARCHAR
        )
        """
    )

    total_rows = 0

    # --------------------------------------------------------
    # Stream TSV.
    # --------------------------------------------------------

    with open(
        input_file,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        batch = []

        for row in reader:

            entity_id = row.get(
                "entity_id",
                ""
            )

            business_name = row.get(
                "business_name",
                ""
            )

            business_address = row.get(
                "business_address",
                ""
            )

            country = row.get(
                "country",
                ""
            )

            batch.append(
                (
                    entity_id,
                    business_name,
                    business_address,
                    country,

                    normalize_text(
                        business_name
                    ),

                    normalize_text(
                        business_address
                    ),

                    normalize_country(
                        country
                    )
                )
            )

            if len(batch) >= CHUNK_SIZE:

                con.executemany(
                    f"""
                    INSERT INTO {table_name}
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch
                )

                total_rows += len(batch)

                print(
                    f"\rRows processed: "
                    f"{total_rows:,}",
                    end="",
                    flush=True
                )

                batch.clear()

        # Final batch.
        if batch:

            con.executemany(
                f"""
                INSERT INTO {table_name}
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch
            )

            total_rows += len(batch)

            batch.clear()

    print()

    print(
        f"Finished: {total_rows:,} rows"
    )

    # --------------------------------------------------------
    # Basic indexes/statistics through ordering.
    # --------------------------------------------------------

    print(
        "Running ANALYZE..."
    )

    con.execute(
        f"ANALYZE {table_name}"
    )

    # --------------------------------------------------------
    # Basic validation.
    # --------------------------------------------------------

    stats = con.execute(
        f"""
        SELECT

            COUNT(*) AS rows,

            COUNT(DISTINCT entity_id)
                AS unique_ids,

            COUNT(*) FILTER (
                WHERE name_norm <> ''
            ) AS nonempty_names,

            COUNT(*) FILTER (
                WHERE address_norm <> ''
            ) AS nonempty_addresses,

            COUNT(*) FILTER (
                WHERE address_norm ~ '[0-9]+'
            ) AS addresses_with_number

        FROM {table_name}
        """
    ).fetchone()

    print()
    print(
        f"Rows              : {stats[0]:,}"
    )

    print(
        f"Unique entity IDs  : {stats[1]:,}"
    )

    print(
        f"Nonempty names     : {stats[2]:,}"
    )

    print(
        f"Nonempty addresses : {stats[3]:,}"
    )

    print(
        f"Addresses w/number : {stats[4]:,}"
    )


# ============================================================
# CREATE GROUND TRUTH
# ============================================================

def build_ground_truth(con):

    print()
    print("=" * 70)
    print("BUILDING TRAIN GROUND TRUTH")
    print("=" * 70)

    gt_file = os.path.join(
        TRAIN_DIR,
        "train_ground_truth.tsv"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE train_ground_truth_raw AS

        SELECT *

        FROM read_csv(
            '{gt_file}',
            delim='\\t',
            header=true,
            auto_detect=true
        )
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE train_true_pairs AS

        SELECT

            source1_entity_id AS s1_id,

            trim(
                unnest(
                    string_split(
                        matched_entity_ids,
                        ','
                    )
                )
            ) AS target_id

        FROM train_ground_truth_raw

        WHERE
            matched_entity_ids IS NOT NULL
            AND trim(matched_entity_ids) <> ''
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM train_true_pairs
        """
    ).fetchone()[0]

    print(
        f"True pairs: {count:,}"
    )


# ============================================================
# BUILD SIMPLE FREQUENCY TABLES
# ============================================================

def build_frequency_tables(con):

    print()
    print("=" * 70)
    print("BUILDING NAME FREQUENCY TABLES")
    print("=" * 70)

    for source in [
        "train_s1",
        "train_s2",
        "train_s3",
        "test_s1",
        "test_s2",
        "test_s3"
    ]:

        print(
            f"Building frequencies for {source}..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {source}_name_freq AS

            SELECT

                name_norm,

                COUNT(*) AS frequency

            FROM {source}

            WHERE
                name_norm <> ''

            GROUP BY name_norm
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {source}_addr_token_freq AS

            SELECT

                token,

                COUNT(*) AS frequency

            FROM (

                SELECT DISTINCT

                    entity_id,

                    unnest(
                        string_split(
                            address_norm,
                            ' '
                        )
                    ) AS token

                FROM {source}

                WHERE
                    address_norm <> ''
            )

            WHERE
                token <> ''
                AND length(token) >= 2

            GROUP BY token
            """
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ER — NORMALIZED DUCKDB V2")
    print("=" * 70)

    print()
    print(
        f"Output DB: {OUTPUT_DB}"
    )

    # --------------------------------------------------------
    # Safety: do not overwrite silently.
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_DB):

        raise RuntimeError(
            f"\nDatabase already exists:\n"
            f"{OUTPUT_DB}\n\n"
            f"Delete it manually if you want to rebuild."
        )

    os.makedirs(
        WORKING_DIR,
        exist_ok=True
    )

    os.makedirs(
        TEMP_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Connection.
    # --------------------------------------------------------

    con = duckdb.connect(
        OUTPUT_DB
    )

    con.execute(
        f"SET memory_limit = '{MEMORY_LIMIT}'"
    )

    con.execute(
        f"SET threads = {THREADS}"
    )

    con.execute(
        f"SET temp_directory = '{TEMP_DIR}'"
    )

    con.execute(
        "SET preserve_insertion_order = false"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    process_tsv(
        con,
        os.path.join(
            TRAIN_DIR,
            "train_source1.tsv"
        ),
        "train_s1"
    )

    process_tsv(
        con,
        os.path.join(
            TRAIN_DIR,
            "train_source2.tsv"
        ),
        "train_s2"
    )

    process_tsv(
        con,
        os.path.join(
            TRAIN_DIR,
            "train_source3.tsv"
        ),
        "train_s3"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    process_tsv(
        con,
        os.path.join(
            TEST_DIR,
            "test_source1.tsv"
        ),
        "test_s1"
    )

    process_tsv(
        con,
        os.path.join(
            TEST_DIR,
            "test_source2.tsv"
        ),
        "test_s2"
    )

    process_tsv(
        con,
        os.path.join(
            TEST_DIR,
            "test_source3.tsv"
        ),
        "test_s3"
    )

    # --------------------------------------------------------
    # Ground truth.
    # --------------------------------------------------------

    build_ground_truth(
        con
    )

    # --------------------------------------------------------
    # Frequencies.
    # --------------------------------------------------------

    build_frequency_tables(
        con
    )

    # --------------------------------------------------------
    # Final checkpoint.
    # --------------------------------------------------------

    print()
    print(
        "Running CHECKPOINT..."
    )

    con.execute(
        "CHECKPOINT"
    )

    # --------------------------------------------------------
    # Database summary.
    # --------------------------------------------------------

    size = os.path.getsize(
        OUTPUT_DB
    )

    print()
    print("=" * 70)
    print("DATABASE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Database size: "
        f"{size / (1024 ** 3):.2f} GB"
    )

    con.close()


if __name__ == "__main__":
    main()
