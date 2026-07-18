"""
Heuristic price-adjustment logic based on detected damage.

There's no dataset mapping "this damage" to "this price impact" - these
weights are a documented, reasoned heuristic (relative real-world repair
cost), not a learned function. See README.md for the rationale.
"""

# Base discount (fraction of base price) per damage category for a single
# detected instance, roughly reflecting relative repair cost.
SEVERITY_WEIGHTS = {
    "tire flat": 0.03,
    "lamp broken": 0.04,
    "scratch": 0.03,
    "dent": 0.05,
    "crack": 0.08,
    "glass shatter": 0.10,
}

# Repeated instances of the same category add diminishing extra discount,
# capped at this multiple of the category's base weight.
DIMINISHING_FACTOR = 0.3
PER_CATEGORY_CAP_MULTIPLIER = 2.0

# Total discount across all categories never exceeds this fraction.
MAX_DISCOUNT = 0.5


def compute_discount(detections):
    """Compute a price discount fraction from a list of detections.

    detections: list of {"class_name": str, "confidence": float}.

    Returns (discount_fraction, breakdown) where breakdown is a list of
    {"category": str, "count": int, "contribution": float}, sorted by
    contribution descending.
    """
    counts = {}
    for detection in detections:
        category = detection["class_name"]
        counts[category] = counts.get(category, 0) + 1

    breakdown = []
    total_discount = 0.0
    for category, count in counts.items():
        weight = SEVERITY_WEIGHTS.get(category, 0.0)
        contribution = weight * (1 + DIMINISHING_FACTOR * (count - 1))
        contribution = min(contribution, weight * PER_CATEGORY_CAP_MULTIPLIER)
        breakdown.append({"category": category, "count": count, "contribution": round(contribution, 4)})
        total_discount += contribution

    total_discount = min(total_discount, MAX_DISCOUNT)
    breakdown.sort(key=lambda item: item["contribution"], reverse=True)

    return round(total_discount, 4), breakdown
