import React, { useRef } from 'react'

// Scatter nodes around the orchestrator (which is centered at 1000, 600)
export const AGENT_POSITIONS = {
  'car_control': { x: 650, y: 560 },
  'navigation': { x: 1020, y: 380 },
  'infotainment': { x: 1200, y: 560 },
  'car_manual': { x: 800, y: 800 },
  'cloud': { x: 1200, y: 800 },
}

export function getDefaultPosition(agent) {
  const pos = AGENT_POSITIONS[agent.id];
  if (pos) return pos;
  return {
    x: 600 + (agent.id.length * 20),
    y: 350 + (agent.name.length * 10)
  };
}

export default function AgentNode({ agent, setAgents, canvasRef, isActive }) {
  const nodeRef = useRef(null)

  const { x: defaultX, y: defaultY } = getDefaultPosition(agent);

  const x = agent.x ?? defaultX
  const y = agent.y ?? defaultY

  const style = {
    left: x + 'px',
    top: y + 'px'
  };

  function onMouseDown(e) {
    e.stopPropagation()
    if (e.button !== 0) return
    e.preventDefault()

    const startMouseX = e.clientX
    const startMouseY = e.clientY
    // Capture the initial position before drag
    const startX = x
    const startY = y

    function onMove(ev) {
      ev.preventDefault()
      const dx = ev.clientX - startMouseX
      const dy = ev.clientY - startMouseY
      const newX = startX + dx
      const newY = startY + dy

      // Update state immediately so the wire and node re-render
      setAgents(prev => prev.map(a => a.id === agent.id ? { ...a, x: newX, y: newY } : a))
    }

    function onUp() {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.userSelect = ''
    }

    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  return (
    <div
      ref={nodeRef}
      onMouseDown={onMouseDown}
      className={`absolute bg-[#1a2236] text-white px-4 py-2 rounded-lg flex items-center gap-3 transition-colors cursor-grab ${isActive
        ? 'border-2 border-green-500 shadow-[0_0_20px_rgba(34,197,94,0.4)]'
        : 'border border-sky-500/30 hover:border-sky-400 shadow-[0_0_15px_rgba(0,163,255,0.1)]'
        }`}
      style={style}
    >
      <div className={`w-2 h-2 rounded-full shrink-0 ${isActive ? 'bg-green-500 animate-pulse shadow-[0_0_10px_rgba(34,197,94,0.8)]' : 'bg-gray-500'}`} />
      <span className="text-xl shrink-0">{agent.emoji}</span>
      <span className="text-sm font-medium tracking-wide whitespace-nowrap">{agent.name}</span>
    </div>
  )
}