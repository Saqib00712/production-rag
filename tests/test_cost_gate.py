import pytest

from app.eval.cost_gate import CostCeilingExceeded, check_cost_ceiling


def test_under_ceiling_passes_silently():
    check_cost_ceiling(0.01, ceiling_usd=0.05)  # no exception


def test_exactly_at_ceiling_passes():
    check_cost_ceiling(0.05, ceiling_usd=0.05)  # boundary: equal is allowed, not "exceeded"


def test_over_ceiling_raises_with_both_numbers_in_the_message():
    with pytest.raises(CostCeilingExceeded) as exc_info:
        check_cost_ceiling(0.12, ceiling_usd=0.05)
    assert exc_info.value.total_cost_usd == 0.12
    assert exc_info.value.ceiling_usd == 0.05
    assert "0.12" in str(exc_info.value) and "0.05" in str(exc_info.value)


def test_zero_cost_never_raises():
    check_cost_ceiling(0.0, ceiling_usd=0.0)
