import re

def post_process(agent_response):
    """
    Remove all links, URLs, and emails in the message.
    Returns only the required fields.
    """
    message = agent_response.get('message', '') or ''
    # Remove URLs
    message = re.sub(r'https?://\S+|www\.\S+', '', message)
    # Remove emails
    message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', message)
    # Optionally, remove extra spaces left
    message = re.sub(r'\s+', ' ', message).strip()
    agent_response['message'] = message

    return agent_response