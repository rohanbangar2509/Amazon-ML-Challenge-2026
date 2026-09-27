import os
import re
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

WORKING_DIR = os.path.join(
    BASE_DIR,
    "working"
)

DB_PATH = os.path.join(
    WORKING_DIR,
    "amazon_er.duckdb"
)

TEMP_DIR = os.path.join(
    BASE_DIR,
    "temp",
    "duckdb"
)

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# ============================================================
# SOURCE FILES
# ============================================================

FILES = {
    "train_s1": os.path.join(
        DATA_DIR,
        "train",
        "train_source1.tsv"
    ),

    "train_s2": os.path.join(
        DATA_DIR,
        "train",
        "train_source2.tsv"
    ),

    "train_s3": os.path.join(
        DATA_DIR,
        "train",
        "train_source3.tsv"
    ),

    "test_s1": os.path.join(
        DATA_DIR,
        "test",
        "test_source1.tsv"
    ),

    "test_s2": os.path.join(
        DATA_DIR,
        "test",
        "test_source2.tsv"
    ),

    "test_s3": os.path.join(
        DATA_DIR,
        "test",
        "test_source3.tsv"
    ),
}


# ============================================================
# NORMALIZATION SQL
# ============================================================

def normalized_name_sql(column):
    """
    Unicode-aware-ish normalization using DuckDB SQL.

    Lowercase + punctuation/whitespace normalization.
    """

    return f"""
        lower(
            regexp_replace(
                regexp_replace(
                    {column},
                    '[^[:alnum:]\\\\s]',
                    ' ',
                    'g'
                ),
                '\\\\s+',
                ' ',
                'g'
            )
        )
    """


def normalized_address_sql(column):

    return f"""
        lower(
            regexp_replace(
                regexp_replace(
                    coalesce({column}, ''),
                    '[^[:alnum:]\\\\s]',
                    ' ',
                    'g'
                ),
                '\\\\s+',
                ' ',
                'g'
            )
        )
    """


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE - DUCKDB DATA LAYER")
    print("=" * 70)

    print()
    print(f"Database : {DB_PATH}")
    print(f"Temp dir : {TEMP_DIR}")

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    print()
    print("Opening DuckDB...")

    con = duckdb.connect(DB_PATH)

    # --------------------------------------------------------
    # Resource configuration
    # --------------------------------------------------------

    print("Configuring DuckDB...")

    con.execute(
        "SET memory_limit = '5GB'"
    )

    con.execute(
        "SET threads = 6"
    )

    con.execute(
        f"SET temp_directory = '{TEMP_DIR}'"
    )

    con.execute(
        "SET preserve_insertion_order = false"
    )

    # --------------------------------------------------------
    # Create source tables
    # --------------------------------------------------------

    for table_name, path in FILES.items():

        print()
        print("-" * 70)
        print(f"Loading {table_name}")
        print("-" * 70)

        print(f"File: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        # ----------------------------------------------------
        # Create raw table
        # ----------------------------------------------------

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv(
                '{path}',
                delim='\\t',
                header=true,
                quote='',
                escape='',
                null_padding=true,
                ignore_errors=false,
                auto_detect=true
            )
            """
        )

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {table_name}
            """
        ).fetchone()[0]

        print(
            f"Rows loaded: {count:,}"
        )

        # ----------------------------------------------------
        # Show schema
        # ----------------------------------------------------

        columns = con.execute(
            f"""
            DESCRIBE {table_name}
            """
        ).fetchall()

        print("Columns:")

        for col in columns:
            print(
                f"  {col[0]} : {col[1]}"
            )

        # ----------------------------------------------------
        # Add normalized columns
        # ----------------------------------------------------

        print()
        print("Adding normalized columns...")

        con.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS name_norm VARCHAR
            """
        )

        con.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS address_norm VARCHAR
            """
        )

        con.execute(
            f"""
            UPDATE {table_name}
            SET
                name_norm =
                    trim(
                        {normalized_name_sql("business_name")}
                    ),

                address_norm =
                    trim(
                        {normalized_address_sql("business_address")}
                    )
            """
        )

        # ----------------------------------------------------
        # Add country normalized
        # ----------------------------------------------------

        con.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS country_norm VARCHAR
            """
        )

        con.execute(
            f"""
            UPDATE {table_name}
            SET country_norm =
                lower(
                    trim(country)
                )
            """
        )

        # ----------------------------------------------------
        # Extract first numeric token
        # ----------------------------------------------------

        con.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS first_number VARCHAR
            """
        )

        con.execute(
            f"""
            UPDATE {table_name}
            SET first_number =
                regexp_extract(
                    coalesce(business_address, ''),
                    '[0-9]+',
                    0
                )
            """
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS rows,

                COUNT(
                    NULLIF(name_norm, '')
                ) AS nonempty_names,

                COUNT(
                    NULLIF(address_norm, '')
                ) AS nonempty_addresses,

                COUNT(
                    NULLIF(first_number, '')
                ) AS rows_with_number,

                COUNT(
                    DISTINCT name_norm
                ) AS unique_names,

                COUNT(
                    DISTINCT address_norm
                ) AS unique_addresses

            FROM {table_name}
            """
        ).fetchone()

        print()
        print("Statistics:")

        print(
            f"  Rows                : {stats[0]:,}"
        )

        print(
            f"  Non-empty names     : {stats[1]:,}"
        )

        print(
            f"  Non-empty addresses : {stats[2]:,}"
        )

        print(
            f"  With number         : {stats[3]:,}"
        )

        print(
            f"  Unique names        : {stats[4]:,}"
        )

        print(
            f"  Unique addresses    : {stats[5]:,}"
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        con.execute(
            "CHECKPOINT"
        )

        print(
            f"{table_name} completed."
        )

    # ========================================================
    # Frequency tables
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING FREQUENCY TABLES")
    print("=" * 70)

    # --------------------------------------------------------
    # Name frequencies
    # --------------------------------------------------------

    for source in [
        "train_s2",
        "train_s3",
        "test_s2",
        "test_s3",
    ]:

        print()
        print(
            f"Building name frequency: {source}"
        )

        freq_table = (
            source
            + "_name_freq"
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {freq_table} AS
            SELECT
                name_norm,
                COUNT(*) AS frequency
            FROM {source}
            WHERE name_norm <> ''
            GROUP BY name_norm
            """
        )

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {freq_table}
            """
        ).fetchone()[0]

        print(
            f"  Unique names: {count:,}"
        )

    # --------------------------------------------------------
    # Address token frequencies
    # --------------------------------------------------------

    print()
    print(
        "Building address token frequency tables..."
    )

    for source in [
        "train_s2",
        "train_s3",
        "test_s2",
        "test_s3",
    ]:

        print(
            f"  {source}"
        )

        freq_table = (
            source
            + "_addr_token_freq"
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {freq_table} AS

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
                WHERE address_norm <> ''
            )

            WHERE
                token <> ''

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
            f"    Tokens: {count:,}"
        )

    # ========================================================
    # Final database statistics
    # ========================================================

    print()
    print("=" * 70)
    print("DATABASE SUMMARY")
    print("=" * 70)

    db_size = os.path.getsize(
        DB_PATH
    )

    print(
        f"Database size: "
        f"{db_size / (1024 ** 3):.2f} GB"
    )

    print()
    print("Tables:")

    tables = con.execute(
        """
        SHOW TABLES
        """
    ).fetchall()

    for table in tables:

        name = table[0]

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {name}
            """
        ).fetchone()[0]

        print(
            f"  {name:<35} "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    con.close()

    print()
    print("=" * 70)
    print("DUCKDB BUILD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Database created at:"
    )

    print(
        DB_PATH
    )


if __name__ == "__main__":
    main()
