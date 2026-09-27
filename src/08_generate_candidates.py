import os
import duckdb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.expanduser(
    "~/amazon_ml_challenge"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "working",
    "amazon_er.duckdb"
)

TEMP_DIR = os.path.join(
    BASE_DIR,
    "temp",
    "duckdb"
)


# ============================================================
# RESOURCE LIMITS
# ============================================================

MEMORY_LIMIT = "5GB"
THREADS = 6


# ============================================================
# BLOCK CONFIGURATION
# ============================================================

# Address token:
# target/source token must occur between 2 and 500 times.
ADDR_MIN_FREQ = 2
ADDR_MAX_FREQ = 500

# Name token:
# deliberately much tighter than our rejected 2-50
# name-token-only experiment.
NAME_MIN_FREQ = 2
NAME_MAX_FREQ = 20


# ============================================================
# CONNECTION
# ============================================================

def connect():

    con = duckdb.connect(DB_PATH)

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

    return con


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth(con):

    print()
    print("=" * 70)
    print("BUILDING GROUND TRUTH TABLE")
    print("=" * 70)

    gt_file = os.path.join(
        BASE_DIR,
        "dataset",
        "extracted",
        "student_resource",
        "dataset",
        "train",
        "train_ground_truth.tsv"
    )

    # Raw GT
    con.execute(
        f"""
        CREATE OR REPLACE TABLE train_ground_truth_raw AS

        SELECT *
        FROM read_csv(
            '{gt_file}',
            delim='\\t',
            header=true,
            quote='',
            escape='',
            auto_detect=true
        )
        """
    )

    # One row per true pair.
    #
    # Some S1 rows may have no matches. Those do not create
    # positive candidate pairs.
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
        f"True S1-S2 pairs: {count:,}"
    )


# ============================================================
# ADDRESS TOKENS
# ============================================================

def build_address_token_tables(con):

    print()
    print("=" * 70)
    print("BUILDING ADDRESS TOKEN TABLES")
    print("=" * 70)

    for source in [
        "train_s1",
        "train_s2"
    ]:

        table = source + "_addr_tokens"

        print()
        print(
            f"Building {table}..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS

            SELECT DISTINCT

                entity_id,
                country_norm,
                token

            FROM (

                SELECT

                    entity_id,
                    country_norm,

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
            """
        )

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {table}
            """
        ).fetchone()[0]

        print(
            f"Rows: {count:,}"
        )

        # Frequency per token.
        freq_table = source + "_addr_token_freq"

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {freq_table} AS

            SELECT
                token,
                COUNT(*) AS frequency

            FROM {table}

            GROUP BY token
            """
        )

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {freq_table}
            """
        ).fetchone()[0]

        print(
            f"Unique tokens: {count:,}"
        )


# ============================================================
# ADDRESS NUMBERS
# ============================================================

def build_address_number_tables(con):

    print()
    print("=" * 70)
    print("BUILDING ADDRESS NUMBER TABLES")
    print("=" * 70)

    for source in [
        "train_s1",
        "train_s2"
    ]:

        table = source + "_addr_numbers"

        print()
        print(
            f"Building {table}..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS

            SELECT DISTINCT

                entity_id,
                country_norm,
                number

            FROM (

                SELECT

                    entity_id,
                    country_norm,

                    unnest(
                        regexp_extract_all(
                            address_norm,
                            '[0-9]+'
                        )
                    ) AS number

                FROM {source}

                WHERE
                    address_norm <> ''
            )

            WHERE
                number <> ''
            """
        )

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {table}
            """
        ).fetchone()[0]

        print(
            f"Rows: {count:,}"
        )


# ============================================================
# NAME TOKENS
# ============================================================

def build_name_token_tables(con):

    print()
    print("=" * 70)
    print("BUILDING NAME TOKEN TABLES")
    print("=" * 70)

    for source in [
        "train_s1",
        "train_s2"
    ]:

        table = source + "_name_tokens"

        print()
        print(
            f"Building {table}..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS

            SELECT DISTINCT

                entity_id,
                country_norm,
                token

            FROM (

                SELECT

                    entity_id,
                    country_norm,

                    unnest(
                        string_split(
                            name_norm,
                            ' '
                        )
                    ) AS token

                FROM {source}

                WHERE
                    name_norm <> ''
            )

            WHERE
                token <> ''
                AND length(token) >= 2
            """
        )

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {table}
            """
        ).fetchone()[0]

        print(
            f"Rows: {count:,}"
        )

        # Token frequency.
        freq_table = source + "_name_token_freq"

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {freq_table} AS

            SELECT
                token,
                COUNT(*) AS frequency

            FROM {table}

            GROUP BY token
            """
        )

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {freq_table}
            """
        ).fetchone()[0]

        print(
            f"Unique tokens: {count:,}"
        )


# ============================================================
# BLOCK A
# EXACT NORMALIZED NAME
# ============================================================

def build_block_name(con):

    print()
    print("=" * 70)
    print("BLOCK A — COUNTRY + EXACT NORMALIZED NAME")
    print("=" * 70)

    con.execute(
        """
        CREATE OR REPLACE TABLE candidate_block_name AS

        SELECT DISTINCT

            s1.entity_id AS s1_id,
            s2.entity_id AS target_id

        FROM train_s1 s1

        INNER JOIN train_s2 s2

            ON s1.country_norm = s2.country_norm
            AND s1.name_norm = s2.name_norm

        WHERE
            s1.name_norm <> ''
            AND s2.name_norm <> ''
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_block_name
        """
    ).fetchone()[0]

    print(
        f"Block A candidates: {count:,}"
    )

    return count


# ============================================================
# BLOCK B
#
# COUNTRY + ADDRESS NUMBER + SELECTIVE ADDRESS TOKEN
#
# Both sides must have token frequency <= 500.
# ============================================================

def build_block_address(con):

    print()
    print("=" * 70)
    print(
        "BLOCK B — COUNTRY + ADDRESS NUMBER + ADDRESS TOKEN"
    )
    print("=" * 70)

    print()
    print("Counting join size before materialization...")

    estimate = con.execute(
        f"""
        SELECT COUNT(*)

        FROM train_s1_addr_numbers n1

        INNER JOIN train_s1_addr_tokens t1

            ON t1.entity_id = n1.entity_id

        INNER JOIN train_s1_addr_token_freq f1

            ON f1.token = t1.token

        INNER JOIN train_s2_addr_numbers n2

            ON n2.number = n1.number

        INNER JOIN train_s2_addr_tokens t2

            ON t2.entity_id = n2.entity_id
            AND t2.token = t1.token

        INNER JOIN train_s2_addr_token_freq f2

            ON f2.token = t2.token

        WHERE

            n1.country_norm = n2.country_norm

            AND f1.frequency >= {ADDR_MIN_FREQ}
            AND f1.frequency <= {ADDR_MAX_FREQ}

            AND f2.frequency >= {ADDR_MIN_FREQ}
            AND f2.frequency <= {ADDR_MAX_FREQ}
        """
    ).fetchone()[0]

    print(
        f"Estimated raw join rows: {estimate:,}"
    )

    # Safety threshold.
    #
    # If the raw join is enormous, do not materialize it.
    if estimate > 80_000_000:

        print()
        print(
            "WARNING: Address block is too large."
        )

        print(
            "Skipping materialization."
        )

        return None

    print()
    print("Materializing Block B...")

    con.execute(
        """
        CREATE OR REPLACE TABLE candidate_block_address AS

        SELECT DISTINCT

            n1.entity_id AS s1_id,
            n2.entity_id AS target_id

        FROM train_s1_addr_numbers n1

        INNER JOIN train_s1_addr_tokens t1

            ON t1.entity_id = n1.entity_id

        INNER JOIN train_s1_addr_token_freq f1

            ON f1.token = t1.token

        INNER JOIN train_s2_addr_numbers n2

            ON n2.number = n1.number

        INNER JOIN train_s2_addr_tokens t2

            ON t2.entity_id = n2.entity_id
            AND t2.token = t1.token

        INNER JOIN train_s2_addr_token_freq f2

            ON f2.token = t2.token

        WHERE

            n1.country_norm = n2.country_norm

            AND f1.frequency >= 2
            AND f1.frequency <= 500

            AND f2.frequency >= 2
            AND f2.frequency <= 500
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_block_address
        """
    ).fetchone()[0]

    print(
        f"Block B candidates: {count:,}"
    )

    return count


# ============================================================
# BLOCK C
#
# COUNTRY + NAME TOKEN + ADDRESS NUMBER
#
# Both name-token frequencies are capped at 2-20.
# ============================================================

def build_block_name_address(con):

    print()
    print("=" * 70)
    print(
        "BLOCK C — COUNTRY + NAME TOKEN + ADDRESS NUMBER"
    )
    print("=" * 70)

    print()
    print("Counting join size before materialization...")

    estimate = con.execute(
        f"""
        SELECT COUNT(*)

        FROM train_s1_name_tokens nt1

        INNER JOIN train_s1_name_token_freq nf1

            ON nf1.token = nt1.token

        INNER JOIN train_s1_addr_numbers an1

            ON an1.entity_id = nt1.entity_id

        INNER JOIN train_s2_name_tokens nt2

            ON nt2.token = nt1.token
            AND nt2.country_norm = nt1.country_norm

        INNER JOIN train_s2_name_token_freq nf2

            ON nf2.token = nt2.token

        INNER JOIN train_s2_addr_numbers an2

            ON an2.entity_id = nt2.entity_id
            AND an2.number = an1.number

        WHERE

            nf1.frequency >= {NAME_MIN_FREQ}
            AND nf1.frequency <= {NAME_MAX_FREQ}

            AND nf2.frequency >= {NAME_MIN_FREQ}
            AND nf2.frequency <= {NAME_MAX_FREQ}
        """
    ).fetchone()[0]

    print(
        f"Estimated raw join rows: {estimate:,}"
    )

    if estimate > 50_000_000:

        print()
        print(
            "WARNING: Name+address block is too large."
        )

        print(
            "Skipping materialization."
        )

        return None

    print()
    print("Materializing Block C...")

    con.execute(
        """
        CREATE OR REPLACE TABLE candidate_block_name_address AS

        SELECT DISTINCT

            nt1.entity_id AS s1_id,
            nt2.entity_id AS target_id

        FROM train_s1_name_tokens nt1

        INNER JOIN train_s1_name_token_freq nf1

            ON nf1.token = nt1.token

        INNER JOIN train_s1_addr_numbers an1

            ON an1.entity_id = nt1.entity_id

        INNER JOIN train_s2_name_tokens nt2

            ON nt2.token = nt1.token
            AND nt2.country_norm = nt1.country_norm

        INNER JOIN train_s2_name_token_freq nf2

            ON nf2.token = nt2.token

        INNER JOIN train_s2_addr_numbers an2

            ON an2.entity_id = nt2.entity_id
            AND an2.number = an1.number

        WHERE

            nf1.frequency >= 2
            AND nf1.frequency <= 20

            AND nf2.frequency >= 2
            AND nf2.frequency <= 20
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_block_name_address
        """
    ).fetchone()[0]

    print(
        f"Block C candidates: {count:,}"
    )

    return count


# ============================================================
# BLOCK EVALUATION
# ============================================================

def evaluate_block(
    con,
    block_table,
    block_name
):

    if block_table is None:
        return

    print()
    print("-" * 70)
    print(
        f"EVALUATING {block_name}"
    )
    print("-" * 70)

    total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {block_table}
        """
    ).fetchone()[0]

    true_recovered = con.execute(
        f"""
        SELECT COUNT(*)

        FROM {block_table} c

        INNER JOIN train_true_pairs gt

            ON gt.s1_id = c.s1_id
            AND gt.target_id = c.target_id
        """
    ).fetchone()[0]

    total_true = con.execute(
        """
        SELECT COUNT(*)
        FROM train_true_pairs
        """
    ).fetchone()[0]

    recall = (
        true_recovered / total_true * 100
        if total_true
        else 0
    )

    print(
        f"Candidates : {total:,}"
    )

    print(
        f"True pairs : {true_recovered:,}"
    )

    print(
        f"Recall     : {recall:.4f}%"
    )

    # Candidate distribution.
    stats = con.execute(
        f"""
        SELECT

            avg(cnt),
            quantile_cont(cnt, 0.50),
            quantile_cont(cnt, 0.90),
            quantile_cont(cnt, 0.95),
            quantile_cont(cnt, 0.99),
            quantile_cont(cnt, 0.999),
            max(cnt)

        FROM (

            SELECT
                s1_id,
                COUNT(*) AS cnt

            FROM {block_table}

            GROUP BY s1_id
        )
        """
    ).fetchone()

    print()
    print("Candidates / S1")

    print(
        f"Mean : {stats[0]:.2f}"
    )

    print(
        f"P50  : {stats[1]:.0f}"
    )

    print(
        f"P90  : {stats[2]:.0f}"
    )

    print(
        f"P95  : {stats[3]:.0f}"
    )

    print(
        f"P99  : {stats[4]:.0f}"
    )

    print(
        f"P99.9: {stats[5]:.0f}"
    )

    print(
        f"MAX  : {stats[6]:,}"
    )


# ============================================================
# UNION
# ============================================================

def build_union(con):

    print()
    print("=" * 70)
    print("BUILDING CONTROLLED UNION")
    print("=" * 70)

    available = []

    for table in [
        "candidate_block_name",
        "candidate_block_address",
        "candidate_block_name_address"
    ]:

        exists = con.execute(
            f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{table}'
            """
        ).fetchone()[0]

        if exists:
            available.append(table)

    if not available:

        raise RuntimeError(
            "No candidate blocks were created."
        )

    union_sql = "\nUNION ALL\n".join(
        [
            f"""
            SELECT
                s1_id,
                target_id
            FROM {table}
            """
            for table in available
        ]
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE candidate_pairs_train_s2 AS

        SELECT DISTINCT
            s1_id,
            target_id

        FROM (
            {union_sql}
        )
        """
    )

    total = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_pairs_train_s2
        """
    ).fetchone()[0]

    true_recovered = con.execute(
        """
        SELECT COUNT(*)

        FROM candidate_pairs_train_s2 c

        INNER JOIN train_true_pairs gt

            ON gt.s1_id = c.s1_id
            AND gt.target_id = c.target_id
        """
    ).fetchone()[0]

    total_true = con.execute(
        """
        SELECT COUNT(*)
        FROM train_true_pairs
        """
    ).fetchone()[0]

    recall = (
        true_recovered / total_true * 100
    )

    print()
    print(
        f"Blocks used          : "
        f"{', '.join(available)}"
    )

    print(
        f"Union candidates      : "
        f"{total:,}"
    )

    print(
        f"True pairs recovered  : "
        f"{true_recovered:,}"
    )

    print(
        f"Union recall          : "
        f"{recall:.4f}%"
    )

    # Distribution.
    stats = con.execute(
        """
        SELECT

            avg(cnt),
            quantile_cont(cnt, 0.50),
            quantile_cont(cnt, 0.90),
            quantile_cont(cnt, 0.95),
            quantile_cont(cnt, 0.99),
            quantile_cont(cnt, 0.999),
            max(cnt)

        FROM (

            SELECT
                s1_id,
                COUNT(*) AS cnt

            FROM candidate_pairs_train_s2

            GROUP BY s1_id
        )
        """
    ).fetchone()

    print()
    print("UNION candidates / S1")

    print(
        f"Mean : {stats[0]:.2f}"
    )

    print(
        f"P50  : {stats[1]:.0f}"
    )

    print(
        f"P90  : {stats[2]:.0f}"
    )

    print(
        f"P95  : {stats[3]:.0f}"
    )

    print(
        f"P99  : {stats[4]:.0f}"
    )

    print(
        f"P99.9: {stats[5]:.0f}"
    )

    print(
        f"MAX  : {stats[6]:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CONTROLLED CANDIDATE GENERATION")
    print("=" * 70)

    print()
    print(
        f"Database: {DB_PATH}"
    )

    con = connect()

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    build_ground_truth(con)

    # --------------------------------------------------------
    # Helper tables
    # --------------------------------------------------------

    build_address_token_tables(con)

    build_address_number_tables(con)

    build_name_token_tables(con)

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    block_a = build_block_name(con)

    block_b = build_block_address(con)

    block_c = build_block_name_address(con)

    # --------------------------------------------------------
    # Evaluate individually
    # --------------------------------------------------------

    evaluate_block(
        con,
        "candidate_block_name",
        "BLOCK A"
    )

    if block_b is not None:

        evaluate_block(
            con,
            "candidate_block_address",
            "BLOCK B"
        )

    if block_c is not None:

        evaluate_block(
            con,
            "candidate_block_name_address",
            "BLOCK C"
        )

    # --------------------------------------------------------
    # Union
    # --------------------------------------------------------

    build_union(con)

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    con.execute(
        "CHECKPOINT"
    )

    con.close()

    print()
    print("=" * 70)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
