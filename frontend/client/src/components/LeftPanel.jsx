
import React, { useState } from 'react'
import { CATALOG } from '../App'
import { addAgent as apiAddAgent, removeAgent as apiRemoveAgent } from '../services/api'

export default function LeftPanel({ agents, setAgents, appendLog }) {
  const [selectedAgentToAdd, setSelectedAgentToAdd] = useState('')

  const unaddedAgents = CATALOG.filter(a => !agents.find(x => x.id === a.id))


  async function addAgent() {
    if (!selectedAgentToAdd) return
    const meta = CATALOG.find(a => a.id === selectedAgentToAdd)
    if (!meta) return
    const newA = { ...meta, x: null, y: null, pinned: false }
    setAgents(a => [...a, newA])
    appendLog(`Added ${newA.name}`)
    setSelectedAgentToAdd('')
    try {
      await apiAddAgent(meta.url || '', meta.id)
    } catch (err) {
      appendLog(`Error adding agent: ${err.message}`)
    }
  }


  async function deleteAgent(id) {
    setAgents(a => a.filter(x => x.id !== id))
    const a = agents.find(x => x.id === id)
    if (a) appendLog(`Deleted ${a.name}`)
    try {
      await apiRemoveAgent(id)
    } catch (err) {
      appendLog(`Error deleting agent: ${err.message}`)
    }
  }

  return (
    <div className="flex flex-col h-full bg-linear-to-b from-[#0f182a] to-[#0e1424] border border-white/5 rounded-xl shadow-[0_10px_30px_rgba(0,0,0,0.35)] p-4 overflow-hidden">
      <div className="font-extrabold text-[#d9e6ff] tracking-wide mb-3 flex items-center gap-2">
        <div className="flex-1">Agent Management</div>
      </div>

      <div className="flex-1 min-h-0 flex flex-col gap-3 overflow-y-auto pr-1 custom-scrollbar">
        <div className="bg-[#0c1220] border border-white/5 rounded-xl p-3 flex flex-col gap-2">
          <label className="text-[#9db0cc] text-sm">New Agent</label>
          <select
            value={selectedAgentToAdd}
            onChange={e => setSelectedAgentToAdd(e.target.value)}
            className="w-full bg-[#0b1322] border border-white/10 text-[#e6eefc] p-2.5 rounded-lg outline-none cursor-pointer"
            disabled={unaddedAgents.length === 0}
          >
            {unaddedAgents.length === 0 ? (
              <option value="">All agents added</option>
            ) : (
              <>
                <option value="">Choose an agent…</option>
                {unaddedAgents.map(a => (
                  <option key={a.id} value={a.id}>{a.name}</option>
                ))}
              </>
            )}
          </select>
          <button
            onClick={addAgent}
            disabled={unaddedAgents.length === 0 || !selectedAgentToAdd}
            className="border-0 p-2.5 rounded-lg font-bold cursor-pointer bg-gradient-to-r from-[#1bb1ff] to-[#6c7bff] text-[#08101e] hover:brightness-110 transition-all disabled:opacity-50 disabled:cursor-not-allowed mt-1 tracking-wide"
          >
            Add Agent
          </button>
          <div className="border-l-2 border-[#3a4a6b] pl-2 py-1 mt-1 bg-[#0c1426] rounded-md text-[#c9d8f6] text-xs">
            Pick from unadded agents. You can re-add any agent you deleted.
          </div>
        </div>

        <div className="font-bold tracking-wide mt-2 text-[#d9e6ff]">Agents</div>
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          {agents.map(a => (
            <div key={a.id} onClick={() => { }} className="flex justify-between items-center p-3 bg-[#0c1527] border border-white/5 rounded-xl hover:bg-[#0f1a30] transition-colors cursor-pointer group">
              <div className="flex items-center gap-2">
                <span className="text-xl">{a.emoji}</span>
                <span className="text-sm font-semibold text-[#dfeaff]">{a.name}</span>
                {a.group && (
                  <span className="text-[10px] text-[#cde7ff] bg-[#142037] border border-white/10 px-2 py-0.5 rounded-full ml-1 whitespace-nowrap">
                    {a.group}
                  </span>
                )}
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); deleteAgent(a.id); }}
                className="w-7 h-7 flex items-center justify-center rounded-lg border border-white/10 bg-[#1b2237] text-red-500 opacity-60 hover:opacity-100 hover:bg-[#ff6b6b]/20 transition-all shrink-0"
                title="Delete agent"
              >
                🗑️
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
