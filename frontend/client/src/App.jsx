import React, { useEffect, useState, useRef } from 'react'
import LeftPanel from './components/LeftPanel'
import Canvas from './components/Canvas'
import RightPanel from './components/RightPanel'
import { getAgents } from './services/api'

export const CATALOG = [
  { id: 'car_control', name: 'Car Control Agent', group: 'Car', emoji: '🛞', url: 'http://car_control:8001' },
  { id: 'navigation', name: 'Navigation Agent', group: 'Navigation', emoji: '🧭', url: 'http://navigation:8003' },
  { id: 'infotainment', name: 'Infotainment Agent', group: 'Media', emoji: '🎵', url: 'http://infotainment:8004' },
  { id: 'car_manual', name: 'Car Manual Agent', group: 'Docs', emoji: '📘', url: 'http://car_manual:8002' },
  { id: 'cloud', name: 'Cloud Agent', group: 'Cloud', emoji: '☁️', url: 'http://cloud:8005' }
];

export default function App() {
  const [agents, setAgents] = useState([])
  const [log, setLog] = useState([])
  const [leftWidth, setLeftWidth] = useState(300)
  const [rightWidth, setRightWidth] = useState(420)
  const [activeAgents, setActiveAgents] = useState([])
  const [isDone, setIsDone] = useState(false)
  const containerRef = useRef(null)

  useEffect(() => {
    // load registered agents from backend server
    const loadAgents = async () => {
      try {
        const backendAgents = await getAgents();
        if (backendAgents && backendAgents.length > 0) {
          const loaded = backendAgents.map(bAgent => {
            const meta = CATALOG.find(a => a.id === bAgent.agent_id) || { id: bAgent.agent_id, name: bAgent.agent_id, group: 'Unknown', emoji: '🤖', url: bAgent.url };
            return { ...meta, x: null, y: null, pinned: false };
          });
          setAgents(loaded);
          appendLog(`Loaded ${loaded.length} agents from orchestrator.`);
        } else {
          setAgents([]); // empty state initially
        }
      } catch (err) {
        appendLog(`Error loading agents: ${err.message}`);
        setAgents([]);
      }
    };

    loadAgents();
  }, [])

  function appendLog(entry) { setLog(l => [...l, entry]) }

  function handleSendCommand(text) {
    appendLog(`User: ${text}`)
    sendCommand(text).then(res => appendLog(`Server: ${JSON.stringify(res)}`)).catch(err => appendLog(`Error: ${err}`))
  }

  // gridTemplateColumns controlled so resizers can update widths
  const gridStyle = { gridTemplateColumns: `${leftWidth}px 1fr ${rightWidth}px` }

  return (
    <div ref={containerRef} style={gridStyle} className="grid gap-4 h-screen p-4 bg-[#0b1020] text-[#e6eefc]">
      <div className="relative flex flex-col h-full min-h-0">
        <LeftPanel agents={agents} setAgents={setAgents} appendLog={appendLog} />
        {/* left resizer */}
        <div
          onMouseDown={(e) => {
            if (e.button !== 0) return
            e.preventDefault()
            const startX = e.clientX
            const startW = leftWidth
            function onMove(ev) {
              const dx = ev.clientX - startX
              const newW = Math.min(800, Math.max(150, startW + dx))
              setLeftWidth(newW)
            }
            function onUp() { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); document.body.style.userSelect = '' }
            document.body.style.userSelect = 'none'
            window.addEventListener('mousemove', onMove)
            window.addEventListener('mouseup', onUp)
          }}
          className="absolute right-0 top-0 bottom-0 w-2 -mr-1 z-30 cursor-col-resize"
        />
      </div>

      <div className="flex flex-col h-full min-h-0">
        <Canvas agents={agents} setAgents={setAgents} appendLog={appendLog} activeAgents={activeAgents} isDone={isDone} />
      </div>

      <div className="relative flex flex-col h-full min-h-0">
        {/* right resizer (left edge of right panel) */}
        <div
          onMouseDown={(e) => {
            if (e.button !== 0) return
            e.preventDefault()
            const startX = e.clientX
            const startW = rightWidth
            function onMove(ev) {
              const dx = startX - ev.clientX
              const newW = Math.min(900, Math.max(160, startW + dx))
              setRightWidth(newW)
            }
            function onUp() { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); document.body.style.userSelect = '' }
            document.body.style.userSelect = 'none'
            window.addEventListener('mousemove', onMove)
            window.addEventListener('mouseup', onUp)
          }}
          className="absolute left-0 top-0 bottom-0 w-2 -ml-1 z-30 cursor-col-resize"
        />
        <RightPanel log={log} setActiveAgents={setActiveAgents} setIsDone={setIsDone} />
      </div>
    </div>
  )
}
