from utils.pricing import MAX_DISCOUNT, SEVERITY_WEIGHTS, compute_discount


def test_no_damage_means_no_discount():
    discount, breakdown = compute_discount([])
    assert discount == 0
    assert breakdown == []


def test_single_instance_uses_base_weight():
    detections = [{"class_name": "dent", "confidence": 0.9}]
    discount, breakdown = compute_discount(detections)
    assert discount == SEVERITY_WEIGHTS["dent"]
    assert breakdown == [{"category": "dent", "count": 1, "contribution": SEVERITY_WEIGHTS["dent"]}]


def test_repeated_instances_diminish_and_cap_per_category():
    # Many scratches shouldn't add unlimited discount - capped at 2x the base weight.
    detections = [{"class_name": "scratch", "confidence": 0.9} for _ in range(10)]
    discount, breakdown = compute_discount(detections)
    assert discount == round(SEVERITY_WEIGHTS["scratch"] * 2, 4)
    assert breakdown[0]["count"] == 10


def test_total_discount_capped_at_max():
    # Max out every category (each capped at 2x its base weight) - sums well
    # past MAX_DISCOUNT, so the overall cap must kick in.
    detections = []
    for category in SEVERITY_WEIGHTS:
        detections += [{"class_name": category, "confidence": 0.9}] * 5
    discount, _ = compute_discount(detections)
    assert discount == MAX_DISCOUNT


def test_unknown_category_contributes_nothing():
    detections = [{"class_name": "totally_unknown", "confidence": 0.9}]
    discount, breakdown = compute_discount(detections)
    assert discount == 0
    assert breakdown == [{"category": "totally_unknown", "count": 1, "contribution": 0.0}]
