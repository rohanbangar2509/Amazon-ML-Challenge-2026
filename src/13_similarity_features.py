import duckdb

DB = "working/amazon_er.duckdb"

con = duckdb.connect(DB)

print("=" * 70)
print("STAGE 13 — STRING SIMILARITY FEATURES")
print("=" * 70)

print("\nCreating compact similarity features...")

con.execute("""
CREATE OR REPLACE TABLE train_s2_similarity AS

SELECT
    s1_id,
    target_id,
    is_match,

    name_exact,
    address_exact,
    country_match,

    s1_name_norm,
    s2_name_norm,
    s1_address_norm,
    s2_address_norm,

    /*
       DuckDB edit distance.
       Convert to normalized similarity in [0,1].
    */
    CASE
        WHEN s1_name_norm = '' OR s2_name_norm = ''
            THEN 0.0
        ELSE
            1.0 -
            (
                levenshtein(s1_name_norm, s2_name_norm)::DOUBLE
                /
                greatest(
                    length(s1_name_norm),
                    length(s2_name_norm),
                    1
                )
            )
    END AS name_edit_sim,

    CASE
        WHEN s1_address_norm = '' OR s2_address_norm = ''
            THEN 0.0
        ELSE
            1.0 -
            (
                levenshtein(s1_address_norm, s2_address_norm)::DOUBLE
                /
                greatest(
                    length(s1_address_norm),
                    length(s2_address_norm),
                    1
                )
            )
    END AS address_edit_sim,

    CASE
        WHEN first_number_s1 IS NOT NULL
         AND first_number_s2 IS NOT NULL
         AND first_number_s1 <> ''
         AND first_number_s1 = first_number_s2
        THEN 1
        ELSE 0
    END AS number_match

FROM (
    SELECT
        f.*,
        s1.first_number AS first_number_s1,
        s2.first_number AS first_number_s2
    FROM train_s2_features f

    INNER JOIN train_s1 s1
        ON f.s1_id = s1.entity_id

    INNER JOIN train_s2 s2
        ON f.target_id = s2.entity_id
) x
""")

n = con.execute("""
SELECT COUNT(*)
FROM train_s2_similarity
""").fetchone()[0]

print(f"Rows created: {n:,}")

print("\nSimilarity statistics by true/non-true pair:")

rows = con.execute("""
SELECT
    is_match,
    COUNT(*) AS pairs,

    ROUND(AVG(name_edit_sim), 4) AS avg_name_sim,
    ROUND(AVG(address_edit_sim), 4) AS avg_address_sim,

    ROUND(
        quantile_cont(name_edit_sim, 0.50),
        4
    ) AS median_name_sim,

    ROUND(
        quantile_cont(address_edit_sim, 0.50),
        4
    ) AS median_address_sim,

    SUM(number_match) AS number_matches

FROM train_s2_similarity

GROUP BY is_match

ORDER BY is_match DESC
""").fetchall()

print(
    f"{'MATCH':>8} "
    f"{'PAIRS':>14} "
    f"{'AVG_NAME':>10} "
    f"{'AVG_ADDR':>10} "
    f"{'MED_NAME':>10} "
    f"{'MED_ADDR':>10} "
    f"{'NUM_MATCH':>12}"
)

print("-" * 85)

for r in rows:
    print(
        f"{r[0]:>8} "
        f"{r[1]:>14,} "
        f"{r[2]:>10} "
        f"{r[3]:>10} "
        f"{r[4]:>10} "
        f"{r[5]:>10} "
        f"{r[6]:>12,}"
    )

print("\n" + "=" * 70)
print("STAGE 13 COMPLETE")
print("=" * 70)

con.close()
