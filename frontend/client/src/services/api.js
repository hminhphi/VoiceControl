const BASE_URL = '/api'

// Add agent to backend
export async function addAgent(url, agentId) {
  const res = await fetch(`${BASE_URL}/agents`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, agent_id: agentId })
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({}))
    throw new Error(error.detail || `HTTP ${res.status}`)
  }
  return await res.json()
}

// Remove agent from backend
export async function removeAgent(agentId) {
  const res = await fetch(`${BASE_URL}/agents/${agentId}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' }
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({}))
    throw new Error(error.detail || `HTTP ${res.status}`)
  }
  return await res.json()
}

// Get all agents from backend
export async function getAgents() {
  const res = await fetch(`${BASE_URL}/agents`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({}))
    throw new Error(error.detail || `HTTP ${res.status}`)
  }
  return await res.json()
}
