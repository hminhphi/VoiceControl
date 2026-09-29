import React, { useRef, useState } from 'react'
import Orchestrator from './Orchestrator'
import AgentNode from './AgentNode'
import WireOverlay from './WireOverlay'


export default function Canvas({ agents, setAgents, appendLog, activeAgents = [] }) {
  const containerRef = useRef(null)
  const innerRef = useRef(null)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const panState = useRef({ dragging: false, startX: 0, startY: 0, startPanX: 0, startPanY: 0 })

  // virtual workspace size (centered in the canvas); panning is clamped so workspace stays in middle zone
  const WORKSPACE_W = 2000
  const WORKSPACE_H = 1200

  function clampPan(x, y) {
    const container = containerRef.current
    if (!container) return { x, y }
    const rect = container.getBoundingClientRect()
    const maxX = Math.max(0, (WORKSPACE_W - rect.width) / 2)
    const maxY = Math.max(0, (WORKSPACE_H - rect.height) / 2)
    const cx = Math.max(-maxX, Math.min(maxX, x))
    const cy = Math.max(-maxY, Math.min(maxY, y))
    return { x: cx, y: cy }
  }

  function onBackgroundMouseDown(e) {
    // Only start panning when clicking the blank area (we rely on node handlers to stopPropagation)
    if (e.button !== 0) return
    e.preventDefault()
    panState.current.dragging = true
    panState.current.startX = e.clientX
    panState.current.startY = e.clientY
    panState.current.startPanX = pan.x
    panState.current.startPanY = pan.y

    function onMove(ev) {
      if (!panState.current.dragging) return
      const dx = ev.clientX - panState.current.startX
      const dy = ev.clientY - panState.current.startY
      const candidate = { x: panState.current.startPanX + dx, y: panState.current.startPanY + dy }
      setPan(clampPan(candidate.x, candidate.y))
    }

    function onUp() {
      panState.current.dragging = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.userSelect = ''
    }

    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  return (
    <main className="bg-linear-to-b from-[#0e1726] to-[#0b1322] rounded-xl p-4 flex-1 min-h-0 flex flex-col">
      <div ref={containerRef} className="relative flex-1 min-h-0 overflow-hidden" onMouseDown={onBackgroundMouseDown}>
        {/* workspace is a large centered area; we translate it by pan (but clamp so it stays around center) */}
        <div
          ref={innerRef}
          className="absolute"
          style={{
            width: WORKSPACE_W + 'px',
            height: WORKSPACE_H + 'px',
            left: '50%',
            top: '50%',
            transform: `translate(calc(-50% + ${pan.x}px), calc(-50% + ${pan.y}px))`
          }}
        >
          <WireOverlay agents={agents} activeAgents={activeAgents} />
          <Orchestrator />
          {agents.map(a => (
            <AgentNode key={a.id} agent={a} setAgents={setAgents} canvasRef={innerRef} isActive={activeAgents.includes(a.id)} />
          ))}
        </div>
      </div>
    </main>
  )
}
