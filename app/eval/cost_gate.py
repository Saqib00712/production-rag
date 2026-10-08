"""
A cost ceiling check for automated (CI) eval runs.

Why this exists as a separate, pure function rather than inline in a
script: CI runs unattended, on a schedule or on every push. A bug anywhere
upstream (a golden set that accidentally grew to 10,000 questions, a PDF
fixture that became enormous, a misconfigured retry loop) should never be
able to turn an automated pipeline into an uncapped OpenAI bill. This is
the single, testable chokepoint that enforces a hard ceiling regardless of
what produced the cost.
"""


class CostCeilingExceeded(Exception):
    def __init__(self, total_cost_usd: float, ceiling_usd: float):
        self.total_cost_usd = total_cost_usd
        self.ceiling_usd = ceiling_usd
        super().__init__(
            f"Cost ceiling exceeded: spent ${total_cost_usd:.6f}, ceiling is ${ceiling_usd:.6f}. "
            f"The run is stopped here deliberately -- a CI pipeline must never have an uncapped OpenAI bill."
        )


def check_cost_ceiling(total_cost_usd: float, ceiling_usd: float) -> None:
    if total_cost_usd > ceiling_usd:
        raise CostCeilingExceeded(total_cost_usd, ceiling_usd)
