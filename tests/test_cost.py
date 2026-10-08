from app.services.cost import estimate_cost_usd


def test_cost_math():
    assert estimate_cost_usd(1_000_000, 0.02) == 0.02
    assert estimate_cost_usd(1_000, 0.02) == 0.00002
    assert estimate_cost_usd(0, 0.02) == 0
