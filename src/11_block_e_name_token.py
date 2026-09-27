import os
import duckdb


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

MEMORY_LIMIT = "5GB"
THREADS = 6

# Target-side name token frequency.
MIN_FREQ = 2
MAX_FREQ = 50

# Hard cap on additional candidates per S1.
MAX_PER_S1 = 50


def connect():

    con = duckdb.connect(DB_PATH)

    con.execute(
        f"SET memory_limit='{MEMORY_LIMIT}'"
    )

    con.execute(
        f"SET threads={THREADS}"
    )

    con.execute(
        f"SET temp_directory='{TEMP_DIR}'"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    return con


def main():

    print("=" * 70)
    print("BLOCK E — SELECTIVE NAME TOKEN")
    print("=" * 70)

    con = connect()

    # --------------------------------------------------------
    # Ground truth must already exist.
    # --------------------------------------------------------

    gt_count = con.execute(
        """
        SELECT COUNT(*)
        FROM train_true_pairs
        """
    ).fetchone()[0]

    print(
        f"Ground-truth pairs: {gt_count:,}"
    )

    # --------------------------------------------------------
    # Build S1 name tokens.
    # --------------------------------------------------------

    print()
    print("Building S1 name tokens...")

    con.execute(
        """
        CREATE OR REPLACE TABLE tmp_s1_name_tokens_e AS

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

            FROM train_s1

            WHERE name_norm <> ''
        )

        WHERE

            token <> ''
            AND length(token) >= 2

            AND token NOT IN (
                'the',
                'and',
                'of',
                'at',
                'in',
                'for',
                'to',
                'llc',
                'inc',
                'corp',
                'corporation',
                'company',
                'co',
                'limited',
                'ltd',
                'plc',
                'llp',
                'pvt',
                'private',
                'group',
                'services',
                'service',
                'center',
                'centre',
                'medical',
                'health',
                'healthcare',
                'hospital',
                'clinic',
                'care',
                'dental',
                'pharmacy',
                'therapy',
                'therapies',
                'practice',
                'practices',
                'general',
                'primary',
                'specialty',
                'specialist',
                'professional',
                'professionals'
            )
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM tmp_s1_name_tokens_e
        """
    ).fetchone()[0]

    print(
        f"S1 name-token rows: {count:,}"
    )

    # --------------------------------------------------------
    # Target-side selective tokens.
    #
    # We use the target token frequency already created by
    # production_pipeline.
    # --------------------------------------------------------

    print()
    print("Building selective target tokens...")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tmp_s2_name_tokens_e AS

        SELECT

            t.entity_id,
            t.country_norm,
            t.token,
            f.frequency

        FROM tmp_train_s2_name_tokens t

        INNER JOIN tmp_train_s2_name_freq f

            ON f.token = t.token

        WHERE

            f.frequency BETWEEN
                {MIN_FREQ}
                AND
                {MAX_FREQ}
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM tmp_s2_name_tokens_e
        """
    ).fetchone()[0]

    print(
        f"Selective target token rows: {count:,}"
    )

    # --------------------------------------------------------
    # Generate raw candidates.
    #
    # Because target frequency <= 50, this is bounded.
    # --------------------------------------------------------

    print()
    print("Generating raw Block E candidates...")

    con.execute(
        """
        CREATE OR REPLACE TABLE tmp_block_e_raw AS

        SELECT DISTINCT

            s1.entity_id AS s1_id,
            t.entity_id AS target_id,
            t.frequency AS token_frequency

        FROM tmp_s1_name_tokens_e s1

        INNER JOIN tmp_s2_name_tokens_e t

            ON t.country_norm = s1.country_norm
            AND t.token = s1.token
        """
    )

    raw_count = con.execute(
        """
        SELECT COUNT(*)
        FROM tmp_block_e_raw
        """
    ).fetchone()[0]

    print(
        f"Raw Block E candidates: {raw_count:,}"
    )

    # --------------------------------------------------------
    # Rank candidates per S1.
    #
    # Lower token frequency = more selective token.
    # --------------------------------------------------------

    print()
    print(
        f"Applying maximum {MAX_PER_S1} candidates per S1..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE candidate_block_e AS

        SELECT

            s1_id,
            target_id

        FROM (

            SELECT

                s1_id,
                target_id,

                ROW_NUMBER() OVER (

                    PARTITION BY s1_id

                    ORDER BY
                        token_frequency ASC,
                        target_id

                ) AS rn

            FROM tmp_block_e_raw
        )

        WHERE rn <= {MAX_PER_S1}
        """
    )

    final_count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_block_e
        """
    ).fetchone()[0]

    print(
        f"Capped Block E candidates: {final_count:,}"
    )

    # --------------------------------------------------------
    # Block E standalone recall.
    # --------------------------------------------------------

    recovered_e = con.execute(
        """
        SELECT COUNT(*)

        FROM candidate_block_e c

        INNER JOIN train_true_pairs gt

            ON gt.s1_id = c.s1_id
            AND gt.target_id = c.target_id
        """
    ).fetchone()[0]

    recall_e = (
        recovered_e / gt_count * 100
    )

    print()
    print("-" * 70)
    print("BLOCK E RESULTS")
    print("-" * 70)

    print(
        f"Candidates recovered: {recovered_e:,}"
    )

    print(
        f"Standalone recall: {recall_e:.4f}%"
    )

    # --------------------------------------------------------
    # Incremental recall over existing candidate set.
    # --------------------------------------------------------

    print()
    print(
        "Calculating incremental recall..."
    )

    old_count = con.execute(
        """
        SELECT COUNT(*)
        FROM train_candidates_s2
        """
    ).fetchone()[0]

    old_recovered = con.execute(
        """
        SELECT COUNT(*)

        FROM train_true_pairs gt

        INNER JOIN train_candidates_s2 c

            ON gt.s1_id = c.s1_id
            AND gt.target_id = c.target_id
        """
    ).fetchone()[0]

    new_recovered = con.execute(
        """
        SELECT COUNT(*)

        FROM train_true_pairs gt

        INNER JOIN (

            SELECT
                s1_id,
                target_id

            FROM train_candidates_s2

            UNION

            SELECT
                s1_id,
                target_id

            FROM candidate_block_e

        ) c

            ON gt.s1_id = c.s1_id
            AND gt.target_id = c.target_id
        """
    ).fetchone()[0]

    incremental = (
        new_recovered - old_recovered
    )

    union_recall = (
        new_recovered / gt_count * 100
    )

    union_count = con.execute(
        """
        SELECT COUNT(*)

        FROM (

            SELECT
                s1_id,
                target_id

            FROM train_candidates_s2

            UNION

            SELECT
                s1_id,
                target_id

            FROM candidate_block_e
        )
        """
    ).fetchone()[0]

    print()
    print("=" * 70)
    print("INCREMENTAL BLOCK E RESULT")
    print("=" * 70)

    print(
        f"Existing candidates : {old_count:,}"
    )

    print(
        f"Existing true pairs : {old_recovered:,}"
    )

    print(
        f"New E true pairs    : {new_recovered:,}"
    )

    print(
        f"Incremental pairs   : {incremental:,}"
    )

    print(
        f"New union candidates: {union_count:,}"
    )

    print(
        f"New union recall    : {union_recall:.4f}%"
    )

    # --------------------------------------------------------
    # Keep Block E for next stage.
    # --------------------------------------------------------

    con.execute(
        "CHECKPOINT"
    )

    con.close()

    print()
    print("=" * 70)
    print("BLOCK E COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
