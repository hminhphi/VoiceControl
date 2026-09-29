import React from 'react'

export default function Orchestrator(){
  return (
    <div id="orchestrator" className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
      <div className="relative group">
        {/* Animated Glow Effect */}
        <div className="absolute -inset-1 bg-linear-to-r from-sky-600 to-indigo-600 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
        
        <div className="relative w-48 h-32 bg-[#0f172a] border border-sky-500/50 rounded-2xl flex flex-col items-center justify-center p-4 text-center">
          <div className="text-sky-400 text-xs font-bold tracking-[0.2em] mb-2 uppercase">Core System</div>
          <div className="font-black text-xs leading-tight text-white">MULTI‑AGENT<br/>ORCHESTRATOR</div>
        </div>
      </div>
    </div>
  )
}
