import React from 'react'
import OrchestratorChat from './OrchestratorChat'

export default function RightPanel({ log, setActiveAgents, setIsDone }) {
  return (
    <aside className="bg-transparent h-full overflow-hidden flex flex-col relative">
      <OrchestratorChat log={log} setActiveAgents={setActiveAgents} setIsDone={setIsDone} />
    </aside>
  )
}
