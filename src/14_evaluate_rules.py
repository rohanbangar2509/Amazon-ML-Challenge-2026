import duckdb

DB = "working/amazon_er.duckdb"

con = duckdb.connect(DB)

print("=" * 80)
print("STAGE 14 — MATCH RULE EVALUATION")
print("=" * 80)

# ------------------------------------------------------------
# Helper: evaluate a rule
# ------------------------------------------------------------

def evaluate(name, condition):
    row = con.execute(f"""
        SELECT
            COUNT(*) AS selected,
            SUM(is_match) AS true_positive
        FROM train_s2_similarity
        WHERE {condition}
    """).fetchone()

    selected = row[0]
    tp = row[1] or 0

    fp = selected - tp

    precision = tp / selected if selected else 0.0

    # Recall relative to ALL S2 ground-truth pairs
    total_true = 3693619

    recall = tp / total_true if total_true else 0.0

    f05 = (
        1.25 * precision * recall /
        (0.25 * precision + recall)
        if precision > 0 and recall > 0
        else 0.0
    )

    print(
        f"{name:<42} "
        f"{selected:>12,} "
        f"{tp:>12,} "
        f"{precision:>9.4f} "
        f"{recall:>9.4f} "
        f"{f05:>9.4f}"
    )


print()
print(
    f"{'RULE':<42} "
    f"{'SELECTED':>12} "
    f"{'TP':>12} "
    f"{'PREC':>9} "
    f"{'RECALL':>9} "
    f"{'F0.5':>9}"
)

print("-" * 105)

# ------------------------------------------------------------
# Very high precision rules
# ------------------------------------------------------------

evaluate(
    "Exact name + exact address",
    """
    name_exact = 1
    AND address_exact = 1
    """
)

evaluate(
    "Exact address",
    """
    address_exact = 1
    """
)

evaluate(
    "Exact name + number + name sim >= .80",
    """
    name_exact = 1
    AND number_match = 1
    AND name_edit_sim >= 0.80
    """
)

# ------------------------------------------------------------
# Similarity combinations
# ------------------------------------------------------------

for n in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    for a in [0.60, 0.70, 0.80, 0.90]:
        evaluate(
            f"name >= {n:.2f}, addr >= {a:.2f}",
            f"""
            name_edit_sim >= {n}
            AND address_edit_sim >= {a}
            """
        )

# ------------------------------------------------------------
# Strong combined rules
# ------------------------------------------------------------

evaluate(
    "Name >= .85 AND address >= .75",
    """
    name_edit_sim >= 0.85
    AND address_edit_sim >= 0.75
    """
)

evaluate(
    "Name >= .80 AND address >= .80",
    """
    name_edit_sim >= 0.80
    AND address_edit_sim >= 0.80
    """
)

evaluate(
    "Name >= .75 AND address >= .85",
    """
    name_edit_sim >= 0.75
    AND address_edit_sim >= 0.85
    """
)

evaluate(
    "Name >= .85 AND address >= .85",
    """
    name_edit_sim >= 0.85
    AND address_edit_sim >= 0.85
    """
)

# ------------------------------------------------------------
# Number-aware rules
# ------------------------------------------------------------

evaluate(
    "Name >= .70 + Addr >= .70 + number",
    """
    name_edit_sim >= 0.70
    AND address_edit_sim >= 0.70
    AND number_match = 1
    """
)

evaluate(
    "Name >= .75 + Addr >= .75 + number",
    """
    name_edit_sim >= 0.75
    AND address_edit_sim >= 0.75
    AND number_match = 1
    """
)

evaluate(
    "Name >= .80 + Addr >= .70 + number",
    """
    name_edit_sim >= 0.80
    AND address_edit_sim >= 0.70
    AND number_match = 1
    """
)

evaluate(
    "Name >= .70 + Addr >= .80 + number",
    """
    name_edit_sim >= 0.70
    AND address_edit_sim >= 0.80
    AND number_match = 1
    """
)

evaluate(
    "Name >= .80 + Addr >= .80 + number",
    """
    name_edit_sim >= 0.80
    AND address_edit_sim >= 0.80
    AND number_match = 1
    """
)

print()
print("=" * 80)
print("STAGE 14 COMPLETE")
print("=" * 80)

con.close()
