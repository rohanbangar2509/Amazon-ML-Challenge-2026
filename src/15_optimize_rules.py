import duckdb

DB = "working/amazon_er.duckdb"

con = duckdb.connect(DB)

TOTAL_TRUE = 3_693_619

print("=" * 90)
print("STAGE 15 — COMBINED MATCH RULE EVALUATION")
print("=" * 90)


def evaluate(name, condition):
    row = con.execute(f"""
        SELECT
            COUNT(*) AS selected,
            COALESCE(SUM(is_match), 0) AS tp
        FROM train_s2_similarity
        WHERE {condition}
    """).fetchone()

    selected = row[0]
    tp = row[1]
    fp = selected - tp

    precision = tp / selected if selected else 0.0
    recall = tp / TOTAL_TRUE if TOTAL_TRUE else 0.0

    f05 = (
        1.25 * precision * recall /
        (0.25 * precision + recall)
        if precision > 0 and recall > 0
        else 0.0
    )

    print(
        f"{name:<58} "
        f"{selected:>12,} "
        f"{tp:>12,} "
        f"{precision:>9.4f} "
        f"{recall:>9.4f} "
        f"{f05:>9.4f}"
    )


print()
print(
    f"{'RULE':<58} "
    f"{'SELECTED':>12} "
    f"{'TP':>12} "
    f"{'PREC':>9} "
    f"{'RECALL':>9} "
    f"{'F0.5':>9}"
)

print("-" * 115)

# ------------------------------------------------------------
# Baseline rules
# ------------------------------------------------------------

evaluate(
    "Exact address",
    """
    address_exact = 1
    """
)

evaluate(
    "Name >= .70 AND address >= .60",
    """
    name_edit_sim >= 0.70
    AND address_edit_sim >= 0.60
    """
)

evaluate(
    "Name >= .75 AND address >= .60",
    """
    name_edit_sim >= 0.75
    AND address_edit_sim >= 0.60
    """
)

evaluate(
    "Name >= .80 AND address >= .60",
    """
    name_edit_sim >= 0.80
    AND address_edit_sim >= 0.60
    """
)

# ------------------------------------------------------------
# UNION: exact address + similarity
# ------------------------------------------------------------

evaluate(
    "Exact address OR name >= .70 + address >= .60",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.70
        AND address_edit_sim >= 0.60
    )
    """
)

evaluate(
    "Exact address OR name >= .75 + address >= .60",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.75
        AND address_edit_sim >= 0.60
    )
    """
)

evaluate(
    "Exact address OR name >= .80 + address >= .60",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.80
        AND address_edit_sim >= 0.60
    )
    """
)

# ------------------------------------------------------------
# Add exact-name evidence
# ------------------------------------------------------------

evaluate(
    "Exact address OR exact name + number",
    """
    address_exact = 1
    OR (
        name_exact = 1
        AND number_match = 1
    )
    """
)

evaluate(
    "Exact address OR exact name + number + name sim",
    """
    address_exact = 1
    OR (
        name_exact = 1
        AND number_match = 1
        AND name_edit_sim >= 0.80
    )
    """
)

# ------------------------------------------------------------
# Strong number-aware union
# ------------------------------------------------------------

evaluate(
    "Exact address OR name .70 + addr .60 + number",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.70
        AND address_edit_sim >= 0.60
        AND number_match = 1
    )
    """
)

evaluate(
    "Exact address OR name .75 + addr .60 + number",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.75
        AND address_edit_sim >= 0.60
        AND number_match = 1
    )
    """
)

evaluate(
    "Exact address OR name .80 + addr .60 + number",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.80
        AND address_edit_sim >= 0.60
        AND number_match = 1
    )
    """
)

# ------------------------------------------------------------
# Three-way combinations
# ------------------------------------------------------------

evaluate(
    "Address exact OR name .70+addr .60 OR exact name+number",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.70
        AND address_edit_sim >= 0.60
    )
    OR (
        name_exact = 1
        AND number_match = 1
    )
    """
)

evaluate(
    "Address exact OR name .75+addr .60 OR exact name+number",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.75
        AND address_edit_sim >= 0.60
    )
    OR (
        name_exact = 1
        AND number_match = 1
    )
    """
)

# ------------------------------------------------------------
# Country-aware conservative rule
# ------------------------------------------------------------

evaluate(
    "Address exact OR name .70+addr .60+country",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.70
        AND address_edit_sim >= 0.60
        AND country_match = 1
    )
    """
)

evaluate(
    "Address exact OR name .75+addr .60+country",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.75
        AND address_edit_sim >= 0.60
        AND country_match = 1
    )
    """
)

# ------------------------------------------------------------
# High precision variants
# ------------------------------------------------------------

evaluate(
    "Address exact OR name .80+addr .70",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.80
        AND address_edit_sim >= 0.70
    )
    """
)

evaluate(
    "Address exact OR name .85+addr .60",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.85
        AND address_edit_sim >= 0.60
    )
    """
)

evaluate(
    "Address exact OR name .90+addr .60",
    """
    address_exact = 1
    OR (
        name_edit_sim >= 0.90
        AND address_edit_sim >= 0.60
    )
    """
)

print()
print("=" * 90)
print("STAGE 15 COMPLETE")
print("=" * 90)

con.close()
