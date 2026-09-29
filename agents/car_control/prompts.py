# Structured description for the orchestrator (exposed in agent card description).
# Convention: PURPOSE, INPUT, OUTPUT, ROUTING so the orchestrator can route and build JSON without agent-specific code.

DESCRIPTION = """Control car doors and trunk: open/close the left door, open/close the right door, and open/close the trunk."""
    

def build_orchestrator_description() -> str:
    return f"""PURPOSE: {DESCRIPTION}"""