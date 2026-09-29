import os

def post_process(agent_response):
    # Returns only the required fields
    return {
        'success': agent_response.get('success'),
        'message': agent_response.get('message'),
        'agent_id': agent_response.get('agent_id')
    }