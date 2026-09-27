import duckdb
import time
import os


# ============================================================
# CONFIGURATION
# ============================================================

DB = "working/amazon_er.duckdb"

# Keep this comfortably below the 12 GB WSL memory limit.
MEMORY_LIMIT = "9GB"
THREADS = 6

# Safety limits
MAX_RAW_NAME_TOKEN_BLOCK = 50_000_000
MAX_FINAL_CANDIDATES = 50_000_000

# Selective name-token frequency.
#
# 2 means we ignore unique tokens because exact unique-token
# blocking is already covered by exact-name blocking.
#
# 50 prevents extremely common tokens from exploding.
MAX_NAME_FREQ = 50

# Maximum number of name-token candidates retained per S1.
MAX_NAME_TOKEN_CANDIDATES_PER_S1 = 50


# ============================================================
# CONNECTION
# ============================================================

def connect():
    con = duckdb.connect(DB)

    con.execute(
        f"PRAGMA threads={THREADS}"
    )

    con.execute(
        f"PRAGMA memory_limit='{MEMORY_LIMIT}'"
    )

    return con


# ============================================================
# HELPER: TABLE EXISTS
# ============================================================

def table_exists(con, table_name):
    result = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = ?
        """,
        [table_name]
    ).fetchone()[0]

    return result > 0


# ============================================================
# BUILD NAME HELPERS
# ============================================================

def build_name_helpers(con, target):

    print()
    print("=" * 70)
    print(f"BUILDING NAME HELPERS — {target.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Target name tokens
    # --------------------------------------------------------

    print("Building target name tokens...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        tmp_{target}_name_tokens_test AS

        SELECT DISTINCT
            entity_id,
            token

        FROM
        (
            SELECT
                entity_id,

                unnest(
                    string_split(
                        regexp_replace(
                            lower(name_norm),
                            '[^a-z0-9]+',
                            ' ',
                            'g'
                        ),
                        ' '
                    )
                ) AS token

            FROM {target}

            WHERE name_norm IS NOT NULL
              AND name_norm <> ''
        )

        WHERE length(token) >= 3

          AND token NOT IN
          (
              'the',
              'and',
              'for',
              'with',
              'inc',
              'llc',
              'ltd',
              'limited',
              'corp',
              'corporation',
              'company',
              'co',
              'plc',
              'private',
              'pvt',
              'group'
          )
        """
    )

    target_token_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM tmp_{target}_name_tokens_test
        """
    ).fetchone()[0]

    print(
        f"Target name-token rows: "
        f"{target_token_rows:,}"
    )

    # --------------------------------------------------------
    # Target token frequencies
    # --------------------------------------------------------

    print("Calculating target token frequencies...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        tmp_{target}_name_freq_test AS

        SELECT
            token,
            COUNT(*) AS frequency

        FROM tmp_{target}_name_tokens_test

        GROUP BY token
        """
    )

    # --------------------------------------------------------
    # Selective target tokens
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        tmp_{target}_selective_name_tokens_test AS

        SELECT
            nt.entity_id,
            nt.token,
            nf.frequency

        FROM tmp_{target}_name_tokens_test nt

        INNER JOIN tmp_{target}_name_freq_test nf
            ON nt.token = nf.token

        WHERE nf.frequency BETWEEN 2 AND {MAX_NAME_FREQ}
        """
    )

    selective_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM tmp_{target}_selective_name_tokens_test
        """
    ).fetchone()[0]

    print(
        f"Selective target token rows: "
        f"{selective_rows:,}"
    )


# ============================================================
# BUILD TEST S1 NAME TOKENS
# ============================================================

def build_s1_name_tokens(con):

    print()
    print("Building TEST S1 name tokens...")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        tmp_test_s1_name_tokens_test AS

        SELECT DISTINCT
            entity_id,
            token

        FROM
        (
            SELECT
                entity_id,

                unnest(
                    string_split(
                        regexp_replace(
                            lower(name_norm),
                            '[^a-z0-9]+',
                            ' ',
                            'g'
                        ),
                        ' '
                    )
                ) AS token

            FROM test_s1

            WHERE name_norm IS NOT NULL
              AND name_norm <> ''
        )

        WHERE length(token) >= 3

          AND token NOT IN
          (
              'the',
              'and',
              'for',
              'with',
              'inc',
              'llc',
              'ltd',
              'limited',
              'corp',
              'corporation',
              'company',
              'co',
              'plc',
              'private',
              'pvt',
              'group'
          )
        """
    )

    n = con.execute(
        """
        SELECT COUNT(*)
        FROM tmp_test_s1_name_tokens_test
        """
    ).fetchone()[0]

    print(
        f"S1 name-token rows: {n:,}"
    )


# ============================================================
# BLOCK A — EXACT NAME
# ============================================================

def block_exact_name(con, target):

    print()
    print("Block A — exact normalized name...")

    table = f"tmp_{target}_block_name_test"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS

        SELECT DISTINCT

            s1.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM test_s1 s1

        INNER JOIN {target} t

            ON s1.name_norm = t.name_norm
           AND s1.name_norm <> ''

           AND s1.country_norm = t.country_norm
        """
    )

    n = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    ).fetchone()[0]

    print(
        f"Block A candidates: {n:,}"
    )

    return table


# ============================================================
# BLOCK B — EXACT ADDRESS
# ============================================================

def block_exact_address(con, target):

    print()
    print("Block B — exact normalized address...")

    table = f"tmp_{target}_block_address_test"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS

        SELECT DISTINCT

            s1.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM test_s1 s1

        INNER JOIN {target} t

            ON s1.address_norm = t.address_norm
           AND s1.address_norm <> ''

           AND s1.country_norm = t.country_norm
        """
    )

    n = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    ).fetchone()[0]

    print(
        f"Block B candidates: {n:,}"
    )

    return table


# ============================================================
# BLOCK E — SELECTIVE NAME TOKEN
# ============================================================

def block_name_token(con, target):

    print()
    print("Block E — selective name token...")

    # --------------------------------------------------------
    # Estimate raw join size first.
    # --------------------------------------------------------

    raw_count = con.execute(
        f"""
        SELECT COUNT(*)

        FROM tmp_test_s1_name_tokens_test s1

        INNER JOIN
        tmp_{target}_selective_name_tokens_test t

            ON s1.token = t.token
        """
    ).fetchone()[0]

    print(
        f"Raw Block E candidates: "
        f"{raw_count:,}"
    )

    if raw_count > MAX_RAW_NAME_TOKEN_BLOCK:

        print(
            f"Block E skipped because raw size "
            f"{raw_count:,} exceeds "
            f"{MAX_RAW_NAME_TOKEN_BLOCK:,}"
        )

        return None

    # --------------------------------------------------------
    # Rank target candidates by token rarity.
    #
    # Lower frequency = stronger blocking evidence.
    #
    # Keep at most 50 candidates per S1.
    # --------------------------------------------------------

    table = f"tmp_{target}_block_name_token_test"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS

        SELECT
            s1_id,
            target_id

        FROM
        (
            SELECT

                s1.entity_id AS s1_id,
                t.entity_id AS target_id,
                t.frequency,

                ROW_NUMBER() OVER
                (
                    PARTITION BY s1.entity_id

                    ORDER BY
                        t.frequency ASC,
                        t.entity_id
                ) AS rn

            FROM tmp_test_s1_name_tokens_test s1

            INNER JOIN
            tmp_{target}_selective_name_tokens_test t

                ON s1.token = t.token
        ) ranked

        WHERE rn <= {MAX_NAME_TOKEN_CANDIDATES_PER_S1}
        """
    )

    n = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    ).fetchone()[0]

    print(
        f"Block E candidates after cap: "
        f"{n:,}"
    )

    return table


# ============================================================
# UNION
# ============================================================

def build_union(con, target, blocks):

    print()
    print("=" * 70)
    print(f"BUILDING FINAL TEST CANDIDATES — {target.upper()}")
    print("=" * 70)

    blocks = [
        b for b in blocks
        if b is not None
    ]

    if not blocks:
        raise RuntimeError(
            f"No candidate blocks were generated for {target}"
        )

    union_sql = "\nUNION ALL\n".join(
        f"""
        SELECT
            s1_id,
            target_id
        FROM {b}
        """
        for b in blocks
    )

    output_table = (
        target.replace(
            "test_",
            "test_candidates_",
            1
        )
    )

    # --------------------------------------------------------
    # DISTINCT candidate pairs
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS

        SELECT DISTINCT

            s1_id,
            target_id

        FROM
        (
            {union_sql}
        )
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {output_table}
        """
    ).fetchone()[0]

    print()
    print(
        f"Final {target.upper()} candidates: "
        f"{count:,}"
    )

    if count > MAX_FINAL_CANDIDATES:

        raise RuntimeError(
            f"""
Final candidate set is too large.

Table:
{output_table}

Candidates:
{count:,}

Safety limit:
{MAX_FINAL_CANDIDATES:,}
"""
        )

    # --------------------------------------------------------
    # Candidate distribution
    # --------------------------------------------------------

    stats = con.execute(
        f"""
        SELECT

            COUNT(*) AS s1_count,

            AVG(cnt) AS avg_candidates,

            MEDIAN(cnt) AS median_candidates,

            MAX(cnt) AS max_candidates

        FROM
        (
            SELECT
                s1_id,
                COUNT(*) AS cnt

            FROM {output_table}

            GROUP BY s1_id
        )
        """
    ).fetchone()

    print()
    print("Candidate distribution:")
    print(
        f"S1 entities       : {stats[0]:,}"
    )
    print(
        f"Average candidates: {stats[1]:.2f}"
    )
    print(
        f"Median candidates : {stats[2]:.0f}"
    )
    print(
        f"Maximum candidates: {stats[3]:,}"
    )

    return output_table


# ============================================================
# PROCESS ONE TARGET
# ============================================================

def process_target(con, target):

    start = time.time()

    print()
    print("#" * 70)
    print(f"PROCESSING {target.upper()}")
    print("#" * 70)

    build_name_helpers(
        con,
        target
    )

    build_s1_name_tokens(
        con
    )

    blocks = []

    # --------------------------------------------------------
    # Block A
    # --------------------------------------------------------

    blocks.append(
        block_exact_name(
            con,
            target
        )
    )

    # --------------------------------------------------------
    # Block B
    # --------------------------------------------------------

    blocks.append(
        block_exact_address(
            con,
            target
        )
    )

    # --------------------------------------------------------
    # Block E
    # --------------------------------------------------------

    blocks.append(
        block_name_token(
            con,
            target
        )
    )

    # --------------------------------------------------------
    # Final union
    # --------------------------------------------------------

    build_union(
        con,
        target,
        blocks
    )

    elapsed = time.time() - start

    print()
    print(
        f"{target.upper()} completed in "
        f"{elapsed / 60:.2f} minutes"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(DB):

        raise FileNotFoundError(
            f"DuckDB database not found: {DB}"
        )

    print("=" * 70)
    print("AMAZON ER — TEST CANDIDATE GENERATION")
    print("=" * 70)

    print(
        f"Database : {DB}"
    )

    print(
        f"Threads  : {THREADS}"
    )

    print(
        f"Memory   : {MEMORY_LIMIT}"
    )

    con = connect()

    try:

        # ----------------------------------------------------
        # TEST S2
        # ----------------------------------------------------

        process_target(
            con,
            "test_s2"
        )

        # ----------------------------------------------------
        # Remove S2 temporary helper tables before S3.
        #
        # This reduces memory pressure.
        # ----------------------------------------------------

        print()
        print("Cleaning TEST_S2 temporary tables...")

        con.execute(
            """
            DROP TABLE IF EXISTS
                tmp_test_s2_name_tokens_test
            """
        )

        con.execute(
            """
            DROP TABLE IF EXISTS
                tmp_test_s2_name_freq_test
            """
        )

        con.execute(
            """
            DROP TABLE IF EXISTS
                tmp_test_s2_selective_name_tokens_test
            """
        )

        con.execute(
            """
            DROP TABLE IF EXISTS
                tmp_test_s2_block_name_test
            """
        )

        con.execute(
            """
            DROP TABLE IF EXISTS
                tmp_test_s2_block_address_test
            """
        )

        con.execute(
            """
            DROP TABLE IF EXISTS
                tmp_test_s2_block_name_token_test
            """
        )

        con.execute(
            """
            DROP TABLE IF EXISTS
                tmp_test_s1_name_tokens_test
            """
        )

        # ----------------------------------------------------
        # TEST S3
        # ----------------------------------------------------

        process_target(
            con,
            "test_s3"
        )

        con.execute(
            "CHECKPOINT"
        )

    finally:

        con.close()

    print()
    print("=" * 70)
    print("TEST CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print()
    print("Created:")
    print("  test_candidates_s2")
    print("  test_candidates_s3")


if __name__ == "__main__":
    main()
