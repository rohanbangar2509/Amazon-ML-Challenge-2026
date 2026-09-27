import os
import sys
import duckdb


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.expanduser("~/amazon_ml_challenge")

DB_PATH = os.path.join(
    BASE_DIR,
    "working",
    "amazon_er.duckdb"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

TEMP_DIR = os.path.join(
    BASE_DIR,
    "temp",
    "duckdb"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

MEMORY_LIMIT = "5GB"
THREADS = 6

# Hard safety limits.
MAX_BLOCK = 35_000_000
MAX_FINAL_CANDIDATES = 45_000_000


# ============================================================
# CONNECTION
# ============================================================

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


# ============================================================
# CHECK TABLE
# ============================================================

def table_exists(con, name):

    return con.execute(
        f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name='{name}'
        """
    ).fetchone()[0] > 0


# ============================================================
# GROUND TRUTH
# ============================================================

def build_gt(con):

    if table_exists(con, "train_true_pairs"):
        return

    gt_file = os.path.join(
        BASE_DIR,
        "dataset",
        "extracted",
        "student_resource",
        "dataset",
        "train",
        "train_ground_truth.tsv"
    )

    print("Building training ground truth...")

    con.execute(
        f"""
        CREATE TABLE train_true_pairs AS

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

        FROM read_csv(
            '{gt_file}',
            delim='\\t',
            header=true,
            auto_detect=true
        )

        WHERE
            matched_entity_ids IS NOT NULL
            AND trim(matched_entity_ids) <> ''
        """
    )


# ============================================================
# TARGET-SPECIFIC HELPERS
# ============================================================

def build_helpers(con, target):

    print()
    print("=" * 70)
    print(f"BUILDING HELPERS FOR {target.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Address numbers
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tmp_{target}_numbers AS

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

            FROM {target}

            WHERE address_norm <> ''
        )

        WHERE number <> ''
        """
    )

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tmp_{target}_addr_tokens AS

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

            FROM {target}

            WHERE address_norm <> ''
        )

        WHERE
            token <> ''
            AND length(token) >= 2
        """
    )

    # --------------------------------------------------------
    # Address token frequencies.
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tmp_{target}_addr_freq AS

        SELECT

            token,
            COUNT(*) AS frequency

        FROM tmp_{target}_addr_tokens

        GROUP BY token
        """
    )

    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    stopwords = (
        "'the','and','of','at','in','for','to',"
        "'llc','inc','corp','corporation','company','co',"
        "'limited','ltd','plc','llp','pvt','private',"
        "'group','services','service','center','centre',"
        "'medical','health','healthcare','hospital','clinic',"
        "'care','dental','pharmacy','therapy','therapies',"
        "'practice','practices','general','primary',"
        "'specialty','specialist','professional','professionals'"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tmp_{target}_name_tokens AS

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

            FROM {target}

            WHERE name_norm <> ''
        )

        WHERE

            token <> ''
            AND length(token) >= 2
            AND token NOT IN ({stopwords})
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tmp_{target}_name_freq AS

        SELECT

            token,
            COUNT(*) AS frequency

        FROM tmp_{target}_name_tokens

        GROUP BY token
        """
    )


# ============================================================
# BLOCK A
# EXACT NAME
# ============================================================

def block_exact_name(con, target):

    print()
    print("BLOCK A — EXACT NAME")

    table = f"tmp_{target}_block_name"

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS

        SELECT DISTINCT

            s1.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM train_s1 s1

        INNER JOIN {target} t

            ON s1.country_norm=t.country_norm
            AND s1.name_norm=t.name_norm

        WHERE

            s1.name_norm <> ''
            AND t.name_norm <> ''
        """
    )

    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"Candidates: {count:,}")

    if count > MAX_BLOCK:
        raise RuntimeError(
            f"Exact-name block exceeded safety limit: {count:,}"
        )

    return table


# ============================================================
# BLOCK B
# EXACT ADDRESS
# ============================================================

def block_exact_address(con, target):

    print()
    print("BLOCK B — EXACT ADDRESS")

    table = f"tmp_{target}_block_address_exact"

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS

        SELECT DISTINCT

            s1.entity_id AS s1_id,
            t.entity_id AS target_id

        FROM train_s1 s1

        INNER JOIN {target} t

            ON s1.country_norm=t.country_norm
            AND s1.address_norm=t.address_norm

        WHERE

            s1.address_norm <> ''
            AND t.address_norm <> ''
        """
    )

    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"Candidates: {count:,}")

    if count > MAX_BLOCK:
        raise RuntimeError(
            f"Exact-address block exceeded safety limit: {count:,}"
        )

    return table


# ============================================================
# BLOCK C
#
# NAME TOKEN + ADDRESS NUMBER
#
# Target token frequency capped at 20.
# ============================================================

def block_name_number(con, target):

    print()
    print("BLOCK C — NAME TOKEN + ADDRESS NUMBER")

    table = f"tmp_{target}_block_name_number"

    estimate = con.execute(
        f"""
        SELECT COUNT(*)

        FROM train_s1 s1

        INNER JOIN tmp_{target}_name_tokens nt

            ON nt.country_norm=s1.country_norm

        INNER JOIN tmp_{target}_name_freq nf

            ON nf.token=nt.token

        INNER JOIN tmp_{target}_numbers n2

            ON n2.entity_id=nt.entity_id
            AND n2.country_norm=nt.country_norm

        INNER JOIN (

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

                FROM train_s1

                WHERE address_norm <> ''
            )
        ) n1

            ON n1.entity_id=s1.entity_id
            AND n1.country_norm=n2.country_norm
            AND n1.number=n2.number

        WHERE

            nf.frequency BETWEEN 2 AND 20
        """
    ).fetchone()[0]

    print(
        f"Estimated raw rows: {estimate:,}"
    )

    if estimate > MAX_BLOCK:

        print(
            "Block C skipped because it exceeds safety limit."
        )

        return None

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS

        SELECT DISTINCT

            s1.entity_id AS s1_id,
            nt.entity_id AS target_id

        FROM train_s1 s1

        INNER JOIN (

            SELECT DISTINCT

                entity_id,
                country_norm,
                token

            FROM tmp_{target}_name_tokens
        ) nt

            ON nt.country_norm=s1.country_norm

        INNER JOIN tmp_{target}_name_freq nf

            ON nf.token=nt.token

        INNER JOIN tmp_{target}_numbers n2

            ON n2.entity_id=nt.entity_id
            AND n2.country_norm=nt.country_norm

        INNER JOIN (

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

                FROM train_s1

                WHERE address_norm <> ''
            )
        ) n1

            ON n1.entity_id=s1.entity_id
            AND n1.country_norm=n2.country_norm
            AND n1.number=n2.number

        WHERE

            nf.frequency BETWEEN 2 AND 20
        """
    )

    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"Candidates: {count:,}")

    return table


# ============================================================
# BLOCK D
#
# SELECTIVE ADDRESS ANCHOR
#
# We build compact (country, number, token) anchors first.
# ============================================================

def block_address_anchor(con, target):

    print()
    print("BLOCK D — SELECTIVE ADDRESS ANCHOR")

    table = f"tmp_{target}_block_address_anchor"

    # --------------------------------------------------------
    # Target anchors.
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tmp_{target}_addr_anchor AS

        SELECT DISTINCT

            n.entity_id,
            n.country_norm,
            n.number,
            t.token

        FROM tmp_{target}_numbers n

        INNER JOIN tmp_{target}_addr_tokens t

            ON t.entity_id=n.entity_id
            AND t.country_norm=n.country_norm

        INNER JOIN tmp_{target}_addr_freq f

            ON f.token=t.token

        WHERE

            f.frequency BETWEEN 2 AND 100
        """
    )

    target_anchor_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM tmp_{target}_addr_anchor
        """
    ).fetchone()[0]

    print(
        f"Target anchors: {target_anchor_count:,}"
    )

    # --------------------------------------------------------
    # S1 anchors only for tokens existing in target.
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tmp_s1_addr_anchor AS

        SELECT DISTINCT

            n.entity_id,
            n.country_norm,
            n.number,
            t.token

        FROM (

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

                FROM train_s1

                WHERE address_norm <> ''
            )
        ) n

        INNER JOIN (

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

                FROM train_s1

                WHERE address_norm <> ''
            )

            WHERE
                token <> ''
                AND length(token) >= 2
        ) t

            ON t.entity_id=n.entity_id
            AND t.country_norm=n.country_norm

        INNER JOIN tmp_{target}_addr_freq f

            ON f.token=t.token

        WHERE

            f.frequency BETWEEN 2 AND 100
        """
    )

    s1_anchor_count = con.execute(
        """
        SELECT COUNT(*)
        FROM tmp_s1_addr_anchor
        """
    ).fetchone()[0]

    print(
        f"S1 anchors: {s1_anchor_count:,}"
    )

    # --------------------------------------------------------
    # Estimate join.
    # --------------------------------------------------------

    estimate = con.execute(
        f"""
        SELECT COUNT(*)

        FROM tmp_s1_addr_anchor a

        INNER JOIN tmp_{target}_addr_anchor b

            ON a.country_norm=b.country_norm
            AND a.number=b.number
            AND a.token=b.token
        """
    ).fetchone()[0]

    print(
        f"Estimated raw rows: {estimate:,}"
    )

    if estimate > MAX_BLOCK:

        print(
            "Block D skipped because it exceeds safety limit."
        )

        return None

    # --------------------------------------------------------
    # Materialize.
    # --------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS

        SELECT DISTINCT

            a.entity_id AS s1_id,
            b.entity_id AS target_id

        FROM tmp_s1_addr_anchor a

        INNER JOIN tmp_{target}_addr_anchor b

            ON a.country_norm=b.country_norm
            AND a.number=b.number
            AND a.token=b.token
        """
    )

    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(
        f"Candidates: {count:,}"
    )

    return table


# ============================================================
# UNION
# ============================================================

def build_union(con, target, blocks):

    print()
    print("=" * 70)
    print(f"BUILDING FINAL TRAIN CANDIDATES — {target.upper()}")
    print("=" * 70)

    blocks = [
        b for b in blocks
        if b is not None
    ]

    union_sql = "\nUNION ALL\n".join(
        [
            f"""
            SELECT
                s1_id,
                target_id
            FROM {b}
            """
            for b in blocks
        ]
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE train_candidates_{target} AS

        SELECT DISTINCT

            s1_id,
            target_id

        FROM (
            {union_sql}
        )
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM train_candidates_{target}
        """
    ).fetchone()[0]

    print(
        f"Final candidates: {count:,}"
    )

    if count > MAX_FINAL_CANDIDATES:

        raise RuntimeError(
            f"""
Final candidate set is too large:
{count:,}

Safety limit:
{MAX_FINAL_CANDIDATES:,}
"""
        )

    # --------------------------------------------------------
    # Recall.
    # --------------------------------------------------------

    true_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM train_true_pairs
        """
    ).fetchone()[0]

    recovered = con.execute(
        f"""
        SELECT COUNT(*)

        FROM train_true_pairs gt

        INNER JOIN train_candidates_{target} c

            ON gt.s1_id=c.s1_id
            AND gt.target_id=c.target_id
        """
    ).fetchone()[0]

    recall = (
        recovered / true_pairs * 100
        if true_pairs
        else 0
    )

    print(
        f"True pairs recovered: {recovered:,}"
    )

    print(
        f"Candidate recall: {recall:.4f}%"
    )

    return count, recall


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src/10_production_pipeline.py s2"
        )

        print(
            "python src/10_production_pipeline.py s3"
        )

        sys.exit(1)

    target = sys.argv[1].lower()

    if target not in ("s2", "s3"):

        raise ValueError(
            "Target must be s2 or s3"
        )

    target_table = f"train_{target}"

    if not os.path.exists(DB_PATH):

        raise FileNotFoundError(
            DB_PATH
        )

    print("=" * 70)
    print("AMAZON ER — PRODUCTION CANDIDATE GENERATION")
    print("=" * 70)

    print(
        f"Target: {target_table}"
    )

    con = connect()

    build_gt(con)

    build_helpers(
        con,
        target_table
    )

    blocks = []

    blocks.append(
        block_exact_name(
            con,
            target_table
        )
    )

    blocks.append(
        block_exact_address(
            con,
            target_table
        )
    )

    blocks.append(
        block_name_number(
            con,
            target_table
        )
    )

    blocks.append(
        block_address_anchor(
            con,
            target_table
        )
    )

    build_union(
        con,
        target,
        blocks
    )

    con.execute(
        "CHECKPOINT"
    )

    con.close()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
