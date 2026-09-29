import React from 'react'

export default function ZoomControls(){
  return (
    <div className="absolute right-3 bottom-3 inline-flex gap-2 items-center bg-[rgba(10,16,30,0.65)] p-1 rounded-md">
      <button className="w-7 h-7 rounded-md">−</button>
      <div className="w-14 text-center text-[#cfe1ff]">100%</div>
      <button className="w-7 h-7 rounded-md">+</button>
    </div>
  )
}
