import React from 'react'

export default function Toast({ message }){
  if (!message) return null
  return (
    <div className="fixed left-1/2 -translate-x-1/2 bottom-4 bg-[#0e1726] text-[#dfe9ff] px-4 py-2 rounded-full" role="status">{message}</div>
  )
}
