"""Token -> USD accounting. Pure function, trivially testable."""


def estimate_cost_usd(tokens: int, price_per_million_tokens: float) -> float:
    return round(tokens / 1_000_000 * price_per_million_tokens, 8)
