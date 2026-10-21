
"""First live API smoke test - one bounded request."""

from decimal import Decimal

from dotenv import load_dotenv

from src.agent.budget import AgentBudget
from src.agent.openai_client import build_live_client


def main():
    load_dotenv()

    budget = AgentBudget(
        max_llm_calls=1,
        max_tool_calls=1,
        max_output_tokens_per_call=400,
        max_estimated_cost_usd=Decimal("0.01"),
    )

    client = build_live_client(budget)

    print("Sending one live API request...")

    result = client.create_response(
        instructions=(
            "You are a market risk analyst. "
            "Answer in one concise sentence."
        ),
        input=[{
            "role": "user",
            "content": (
                "What does an equity exposure "
                "limit breach mean?"
            ),
        }],
        tools=[],
        max_input_tokens_estimate=1500,
    )

    print("\nModel response:")
    print(result.response.text)

    print("\nUsage:")
    print("Input tokens:", result.response.input_tokens)
    print("Output tokens:", result.response.output_tokens)
    print(
        "Estimated cost (USD):",
        result.estimated_actual_cost_usd,
    )

    print("\nBudget:")
    print(budget.snapshot())

    print("\nAudit:")
    for record in client.audit_log:
        print(record.to_dict())


if __name__ == "__main__":
    main()
