# Structured description for the orchestrator (exposed in agent card description).

DESCRIPTION = """Answer questions about the car's parts (locations, names, functions) and how to use the car: features, settings, operating procedures, maintenance, warning lights, Bluetooth pairing, and other car‑manual topics. Also answers configured demo Q&A entries about automotive companies and executives."""

def build_orchestrator_description() -> str:
    return f"""DESCRIPTION: {DESCRIPTION}"""


if __name__ == "__main__":
    print("=== Car Manual Agent Orchestrator Description ===\n")
    print(build_orchestrator_description())
