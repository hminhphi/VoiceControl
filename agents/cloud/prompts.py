DESCRIPTION = """This agent answers questions about the world, including news, history, people, places, companies, sports, politics, and statistics.
 Summarize the news, information, and facts from the web to answer the user's question. 
"""

def build_orchestrator_description() -> str:
    return f"""DESCRIPTION: {DESCRIPTION}"""
