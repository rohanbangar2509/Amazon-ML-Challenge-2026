import duckdb

DB = "working/amazon_er.duckdb"

con = duckdb.connect(DB)

print("=" * 70)
print("STAGE 12 — TRAIN MATCHER / FEATURE GENERATION")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Build candidate features
# ------------------------------------------------------------------

print("\nBuilding candidate feature table...")

con.execute("""
CREATE OR REPLACE TABLE train_s2_features AS

WITH candidates AS (
    SELECT
        s1_id,
        target_id
    FROM candidate_pairs_train_s2_final
),

base AS (
    SELECT
        c.s1_id,
        c.target_id,

        s1.business_name AS s1_name,
        s2.business_name AS s2_name,

        s1.business_address AS s1_address,
        s2.business_address AS s2_address,

        s1.country AS s1_country,
        s2.country AS s2_country,

        s1.name_norm AS s1_name_norm,
        s2.name_norm AS s2_name_norm,

        s1.address_norm AS s1_address_norm,
        s2.address_norm AS s2_address_norm

    FROM candidates c

    INNER JOIN train_s1 s1
        ON c.s1_id = s1.entity_id

    INNER JOIN train_s2 s2
        ON c.target_id = s2.entity_id
),

features AS (
    SELECT
        *,

        -- Exact normalized name
        CASE
            WHEN s1_name_norm = s2_name_norm
             AND s1_name_norm <> ''
            THEN 1 ELSE 0
        END AS name_exact,

        -- Exact normalized address
        CASE
            WHEN s1_address_norm = s2_address_norm
             AND s1_address_norm <> ''
            THEN 1 ELSE 0
        END AS address_exact,

        -- Country agreement
        CASE
            WHEN lower(trim(s1_country)) =
                 lower(trim(s2_country))
            THEN 1 ELSE 0
        END AS country_match,

        -- Name length
        length(s1_name_norm) AS s1_name_len,
        length(s2_name_norm) AS s2_name_len,

        -- Address length
        length(s1_address_norm) AS s1_addr_len,
        length(s2_address_norm) AS s2_addr_len,

        -- Name character overlap approximation
        CASE
            WHEN s1_name_norm <> ''
             AND s2_name_norm <> ''
            THEN
                1.0 -
                (
                    abs(length(s1_name_norm) - length(s2_name_norm))
                    /
                    greatest(
                        length(s1_name_norm),
                        length(s2_name_norm),
                        1
                    )
                )
            ELSE 0.0
        END AS name_length_similarity,

        -- Address character-length similarity
        CASE
            WHEN s1_address_norm <> ''
             AND s2_address_norm <> ''
            THEN
                1.0 -
                (
                    abs(length(s1_address_norm) - length(s2_address_norm))
                    /
                    greatest(
                        length(s1_address_norm),
                        length(s2_address_norm),
                        1
                    )
                )
            ELSE 0.0
        END AS address_length_similarity

    FROM base
)

SELECT
    f.*,

    CASE
        WHEN gt.s1_id IS NOT NULL THEN 1
        ELSE 0
    END AS is_match

FROM features f

LEFT JOIN train_true_pairs gt
    ON f.s1_id = gt.s1_id
   AND f.target_id = gt.target_id
   AND starts_with(gt.target_id, 'S2-')
""")

n = con.execute("""
SELECT COUNT(*)
FROM train_s2_features
""").fetchone()[0]

print(f"Feature rows: {n:,}")

matches = con.execute("""
SELECT COUNT(*)
FROM train_s2_features
WHERE is_match = 1
""").fetchone()[0]

nonmatches = n - matches

print(f"Positive pairs : {matches:,}")
print(f"Negative pairs : {nonmatches:,}")

# ------------------------------------------------------------------
# 2. Feature summary
# ------------------------------------------------------------------

print("\nFeature statistics:")

rows = con.execute("""
SELECT
    name_exact,
    address_exact,
    country_match,
    COUNT(*) AS pairs,
    SUM(is_match) AS true_matches
FROM train_s2_features
GROUP BY
    name_exact,
    address_exact,
    country_match
ORDER BY
    true_matches DESC
""").fetchall()

print(
    f"{'NAME':>6} "
    f"{'ADDR':>6} "
    f"{'COUNTRY':>8} "
    f"{'PAIRS':>14} "
    f"{'TRUE':>14}"
)

print("-" * 60)

for r in rows:
    print(
        f"{r[0]:>6} "
        f"{r[1]:>6} "
        f"{r[2]:>8} "
        f"{r[3]:>14,} "
        f"{r[4]:>14,}"
    )

print("\n" + "=" * 70)
print("STAGE 12 COMPLETE")
print("=" * 70)

con.close()
