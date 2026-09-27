import duckdb
import time
import os


# ============================================================
# CONFIGURATION
# ============================================================

DB = "working/amazon_er.duckdb"

THREADS = 6
MEMORY_LIMIT = "9GB"

OUTPUT_DIR = "output"


# ============================================================
# DATABASE CONNECTION
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
# SCORE ONE TEST TARGET
# ============================================================

def score_target(con, target, candidate_table):

    print()
    print("=" * 70)
    print(f"SCORING {target.upper()}")
    print("=" * 70)

    start = time.time()

    scored_table = f"{target}_scored"
    matches_table = f"{target}_matches"

    # --------------------------------------------------------
    # Candidate count
    # --------------------------------------------------------

    candidate_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {candidate_table}
        """
    ).fetchone()[0]

    print(
        f"Candidate pairs: {candidate_count:,}"
    )

    # --------------------------------------------------------
    # Build similarity features
    # --------------------------------------------------------

    print()
    print("Calculating similarity features...")
    print("This is the expensive stage.")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {scored_table} AS

        SELECT

            c.s1_id,
            c.target_id,

            -- ------------------------------------------------
            -- Raw normalized fields
            -- ------------------------------------------------

            s1.name_norm AS s1_name,
            t.name_norm AS target_name,

            s1.address_norm AS s1_address,
            t.address_norm AS target_address,

            s1.country_norm AS s1_country,
            t.country_norm AS target_country,

            s1.first_number AS s1_number,
            t.first_number AS target_number,

            -- ------------------------------------------------
            -- Exact name
            -- ------------------------------------------------

            CASE
                WHEN
                    s1.name_norm <> ''
                    AND
                    s1.name_norm = t.name_norm
                THEN 1
                ELSE 0
            END AS name_exact,

            -- ------------------------------------------------
            -- Exact address
            -- ------------------------------------------------

            CASE
                WHEN
                    s1.address_norm <> ''
                    AND
                    s1.address_norm = t.address_norm
                THEN 1
                ELSE 0
            END AS address_exact,

            -- ------------------------------------------------
            -- Country
            -- ------------------------------------------------

            CASE
                WHEN
                    s1.country_norm = t.country_norm
                THEN 1
                ELSE 0
            END AS country_match,

            -- ------------------------------------------------
            -- First address number
            -- ------------------------------------------------

            CASE
                WHEN
                    s1.first_number IS NOT NULL
                    AND
                    s1.first_number <> ''
                    AND
                    s1.first_number = t.first_number
                THEN 1
                ELSE 0
            END AS number_match,

            -- ------------------------------------------------
            -- Normalized name edit similarity
            -- ------------------------------------------------

            CASE

                WHEN
                    s1.name_norm IS NULL
                    OR
                    t.name_norm IS NULL
                    OR
                    s1.name_norm = ''
                    OR
                    t.name_norm = ''

                THEN 0.0

                ELSE

                    1.0
                    -
                    (
                        levenshtein(
                            s1.name_norm,
                            t.name_norm
                        )::DOUBLE
                        /
                        greatest(
                            length(s1.name_norm),
                            length(t.name_norm),
                            1
                        )
                    )

            END AS name_edit_sim,

            -- ------------------------------------------------
            -- Normalized address edit similarity
            -- ------------------------------------------------

            CASE

                WHEN
                    s1.address_norm IS NULL
                    OR
                    t.address_norm IS NULL
                    OR
                    s1.address_norm = ''
                    OR
                    t.address_norm = ''

                THEN 0.0

                ELSE

                    1.0
                    -
                    (
                        levenshtein(
                            s1.address_norm,
                            t.address_norm
                        )::DOUBLE
                        /
                        greatest(
                            length(s1.address_norm),
                            length(t.address_norm),
                            1
                        )
                    )

            END AS address_edit_sim

        FROM {candidate_table} c

        INNER JOIN test_s1 s1
            ON c.s1_id = s1.entity_id

        INNER JOIN {target} t
            ON c.target_id = t.entity_id
        """
    )

    scored_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {scored_table}
        """
    ).fetchone()[0]

    print(
        f"Scored pairs: {scored_count:,}"
    )

    # --------------------------------------------------------
    # Apply final matching rules
    #
    # These were evaluated on training data.
    #
    # Rule 1:
    #     exact normalized address
    #
    # Rule 2:
    #     exact normalized name
    #     + matching first address number
    #
    # Rule 3:
    #     name similarity >= 0.70
    #     + address similarity >= 0.60
    # --------------------------------------------------------

    print()
    print("Applying final matching rules...")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {matches_table} AS

        SELECT DISTINCT

            s1_id,
            target_id

        FROM {scored_table}

        WHERE

            -- Rule 1
            address_exact = 1

            OR

            -- Rule 2
            (
                name_exact = 1
                AND
                number_match = 1
            )

            OR

            -- Rule 3
            (
                name_edit_sim >= 0.70
                AND
                address_edit_sim >= 0.60
            )
        """
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    match_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {matches_table}
        """
    ).fetchone()[0]

    matched_s1 = con.execute(
        f"""
        SELECT COUNT(DISTINCT s1_id)
        FROM {matches_table}
        """
    ).fetchone()[0]

    total_s1 = con.execute(
        """
        SELECT COUNT(*)
        FROM test_s1
        """
    ).fetchone()[0]

    zero_match = total_s1 - matched_s1

    print()
    print("-" * 70)
    print(f"{target.upper()} MATCH RESULTS")
    print("-" * 70)

    print(
        f"Candidate pairs        : {candidate_count:,}"
    )

    print(
        f"Scored pairs           : {scored_count:,}"
    )

    print(
        f"Predicted matches      : {match_count:,}"
    )

    print(
        f"S1 with >=1 match      : {matched_s1:,}"
    )

    print(
        f"S1 with zero matches   : {zero_match:,}"
    )

    print(
        f"Total test S1          : {total_s1:,}"
    )

    elapsed = time.time() - start

    print(
        f"Elapsed time           : {elapsed / 60:.2f} minutes"
    )

    return matches_table


# ============================================================
# BUILD FINAL MATCH TABLE
# ============================================================

def build_final_matches(con):

    print()
    print("=" * 70)
    print("COMBINING S2 + S3 MATCHES")
    print("=" * 70)

    con.execute(
        """
        CREATE OR REPLACE TABLE final_test_matches AS

        SELECT
            s1_id,
            target_id

        FROM test_s2_matches

        UNION

        SELECT
            s1_id,
            target_id

        FROM test_s3_matches
        """
    )

    total_matches = con.execute(
        """
        SELECT COUNT(*)
        FROM final_test_matches
        """
    ).fetchone()[0]

    matched_s1 = con.execute(
        """
        SELECT COUNT(DISTINCT s1_id)
        FROM final_test_matches
        """
    ).fetchone()[0]

    print(
        f"Total predicted pairs : {total_matches:,}"
    )

    print(
        f"S1 entities matched   : {matched_s1:,}"
    )

    return total_matches


# ============================================================
# WRITE candidate_pairs.tsv
# ============================================================

def write_candidate_pairs(con):

    print()
    print("=" * 70)
    print("WRITING candidate_pairs.tsv")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "candidate_pairs.tsv"
    )

    # --------------------------------------------------------
    # Include all candidate pairs from S2 and S3.
    #
    # The source column explicitly identifies the source.
    # --------------------------------------------------------

    con.execute(
        f"""
        COPY
        (
            SELECT
                s1_id,
                'S2' AS source,
                target_id AS candidate_id

            FROM test_candidates_s2

            UNION ALL

            SELECT
                s1_id,
                'S3' AS source,
                target_id AS candidate_id

            FROM test_candidates_s3
        )

        TO '{path}'

        (
            FORMAT CSV,
            DELIMITER '\\t',
            HEADER
        )
        """
    )

    print(
        f"Created: {path}"
    )

    size_mb = os.path.getsize(path) / (
        1024 * 1024
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )


# ============================================================
# WRITE matching_results.tsv
# ============================================================

def write_matching_results(con):

    print()
    print("=" * 70)
    print("WRITING matching_results.tsv")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "matching_results.tsv"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # LEFT JOIN test_s1 guarantees one output row for
    # every test S1 entity.
    #
    # If no match exists:
    #
    #     matches = ''
    #
    # Otherwise:
    #
    #     S2-xxx,S3-yyy,...
    #
    # DISTINCT prevents duplicate target IDs.
    # --------------------------------------------------------

    con.execute(
        f"""
        COPY
        (
            SELECT

                s1.entity_id AS s1_id,

                COALESCE(
                    string_agg(
                        DISTINCT m.target_id,
                        ',' ORDER BY m.target_id
                    ),
                    ''
                ) AS matches

            FROM test_s1 s1

            LEFT JOIN final_test_matches m

                ON s1.entity_id = m.s1_id

            GROUP BY
                s1.entity_id

            ORDER BY
                s1.entity_id
        )

        TO '{path}'

        (
            FORMAT CSV,
            DELIMITER '\\t',
            HEADER
        )
        """
    )

    print(
        f"Created: {path}"
    )

    size_mb = os.path.getsize(path) / (
        1024 * 1024
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )


# ============================================================
# FINAL STATISTICS
# ============================================================

def print_final_statistics(con):

    print()
    print("=" * 70)
    print("FINAL SUBMISSION STATISTICS")
    print("=" * 70)

    total_s1 = con.execute(
        """
        SELECT COUNT(*)
        FROM test_s1
        """
    ).fetchone()[0]

    matched_s1 = con.execute(
        """
        SELECT COUNT(DISTINCT s1_id)
        FROM final_test_matches
        """
    ).fetchone()[0]

    total_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM final_test_matches
        """
    ).fetchone()[0]

    zero_match = total_s1 - matched_s1

    s2_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM final_test_matches
        WHERE starts_with(target_id, 'S2-')
        """
    ).fetchone()[0]

    s3_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM final_test_matches
        WHERE starts_with(target_id, 'S3-')
        """
    ).fetchone()[0]

    print()
    print(
        f"Total test S1 entities : {total_s1:,}"
    )

    print(
        f"S1 with matches        : {matched_s1:,}"
    )

    print(
        f"S1 with no matches     : {zero_match:,}"
    )

    print(
        f"Total predicted pairs  : {total_pairs:,}"
    )

    print(
        f"S2 predicted pairs     : {s2_pairs:,}"
    )

    print(
        f"S3 predicted pairs     : {s3_pairs:,}"
    )

    print()

    # --------------------------------------------------------
    # Check duplicate pairs
    # --------------------------------------------------------

    duplicate_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM
        (
            SELECT
                s1_id,
                target_id,
                COUNT(*) AS cnt

            FROM final_test_matches

            GROUP BY
                s1_id,
                target_id

            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    print(
        f"Duplicate final pairs  : {duplicate_pairs:,}"
    )

    # --------------------------------------------------------
    # Check invalid source IDs
    # --------------------------------------------------------

    invalid_ids = con.execute(
        """
        SELECT COUNT(*)

        FROM final_test_matches m

        LEFT JOIN
        (
            SELECT entity_id
            FROM test_s2

            UNION

            SELECT entity_id
            FROM test_s3
        ) valid

            ON m.target_id = valid.entity_id

        WHERE valid.entity_id IS NULL
        """
    ).fetchone()[0]

    print(
        f"Invalid target IDs     : {invalid_ids:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ER — FINAL TEST MATCHER")
    print("=" * 70)

    if not os.path.exists(DB):

        raise FileNotFoundError(
            f"DuckDB database not found: {DB}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print(f"Database : {DB}")
    print(f"Threads  : {THREADS}")
    print(f"Memory   : {MEMORY_LIMIT}")

    start = time.time()

    con = connect()

    try:

        # ----------------------------------------------------
        # Verify candidate tables exist
        # ----------------------------------------------------

        required_tables = [
            "test_candidates_s2",
            "test_candidates_s3",
        ]

        for table in required_tables:

            exists = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_name = ?
                """,
                [table]
            ).fetchone()[0]

            if not exists:

                raise RuntimeError(
                    f"Required table missing: {table}"
                )

        # ----------------------------------------------------
        # Score S2
        # ----------------------------------------------------

        score_target(
            con,
            "test_s2",
            "test_candidates_s2"
        )

        # ----------------------------------------------------
        # Score S3
        # ----------------------------------------------------

        score_target(
            con,
            "test_s3",
            "test_candidates_s3"
        )

        # ----------------------------------------------------
        # Combine predictions
        # ----------------------------------------------------

        build_final_matches(
            con
        )

        # ----------------------------------------------------
        # Write required files
        # ----------------------------------------------------

        write_candidate_pairs(
            con
        )

        write_matching_results(
            con
        )

        # ----------------------------------------------------
        # Statistics / validation
        # ----------------------------------------------------

        print_final_statistics(
            con
        )

        con.execute(
            "CHECKPOINT"
        )

    finally:

        con.close()

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("FINAL MATCHING COMPLETE")
    print("=" * 70)

    print(
        f"Total runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("Output files:")

    print(
        "  output/matching_results.tsv"
    )

    print(
        "  output/candidate_pairs.tsv"
    )


if __name__ == "__main__":
    main()
