import React, { useEffect, useRef } from 'react'
import { getDefaultPosition } from './AgentNode'

export default function WireOverlay({ agents, activeAgents = [], isDone = false }) {
  const svgRef = useRef(null)
  const defsRef = useRef(false)

  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return

    // ... (rest of gradient logic remains)
    if (!defsRef.current) {
      const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs')
      const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient')
      gradient.setAttribute('id', 'line-gradient')
      gradient.setAttribute('x1', '0%')
      gradient.setAttribute('y1', '0%')
      gradient.setAttribute('x2', '100%')
      gradient.setAttribute('y2', '100%')

      const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop')
      stop1.setAttribute('offset', '0%')
      stop1.setAttribute('stop-color', '#0ea5e9')
      stop1.setAttribute('stop-opacity', '0.8')

      const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop')
      stop2.setAttribute('offset', '100%')
      stop2.setAttribute('stop-color', '#6366f1')
      stop2.setAttribute('stop-opacity', '0.4')

      gradient.appendChild(stop1)
      gradient.appendChild(stop2)
      defs.appendChild(gradient)
      svg.appendChild(defs)
      defsRef.current = true
    }

    // clear existing paths (keep defs)
    Array.from(svg.querySelectorAll('path')).forEach(p => p.remove())

    // orchestrator is at center of workspace (parent inner div)
    const svgRect = svg.getBoundingClientRect()

    let cx = svgRect.width / 2
    let cy = svgRect.height / 2

    agents.forEach(a => {
      const { x: defaultX, y: defaultY } = getDefaultPosition(a);
      const ex = (a.x ?? defaultX)
      const ey = (a.y ?? defaultY)

      const isActive = activeAgents.includes(a.id)

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
      // cubic bezier from orchestrator center to agent node
      const d = `M ${cx} ${cy} C ${cx} ${(cy + ey) / 2}, ${ex} ${(cy + ey) / 2}, ${ex} ${ey}`

      path.setAttribute('d', d)

      const pulseColor = isDone ? '#6b7280' : (isActive ? '#22c55e' : 'url(#line-gradient)');
      path.setAttribute('stroke', pulseColor)
      path.setAttribute('stroke-width', isActive ? '4' : '2')
      path.setAttribute('fill', 'none')
      path.setAttribute('opacity', isActive ? '1' : '0.6')
      if (isActive && !isDone) {
        path.classList.add('animate-pulse')
      }
      svg.appendChild(path)
    })
  }, [agents, activeAgents, isDone])

  return <svg ref={svgRef} className="absolute inset-0 w-full h-full pointer-events-none" />
}
