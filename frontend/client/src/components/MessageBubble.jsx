import React from "react"
import { CATALOG } from "../App"
import { AGENT_COLORS } from "../utils/chatUtils"
import { getYoutubeId } from "../utils/mediaUtils"

const formatAgents = (agents) => {
    if (!agents || agents.length === 0) return <span className="text-slate-400 font-semibold px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700">Agents</span>;
    return (
        <span className="flex gap-1.5 items-center pl-1">
            {agents.map((a, i) => {
                const colorClass = AGENT_COLORS[a] || 'text-white bg-slate-700 border-slate-600';
                const meta = CATALOG.find(cat => cat.id === a);
                const emoji = meta ? meta.emoji + " " : "";
                return (
                    <span key={a} className={`px-2 py-[2px] rounded-sm tracking-wide border text-[10px] font-bold ${colorClass}`}>
                        {emoji}{a}
                    </span>
                )
            })}
        </span>
    );
}

const formatSystemLog = (text) => {
    if (typeof text !== 'string') return text;

    let foundAgent = null;
    let agentColor = 'text-indigo-400';

    for (const a of CATALOG) {
        if (text.includes(a.name) || text.includes(a.id)) {
            foundAgent = a;
            agentColor = AGENT_COLORS[a.id]?.split(' ')[0] || 'text-indigo-400';
            break;
        }
    }

    const colorizeKeywords = (str) => {
        if (str.includes('Added')) return <span className="text-emerald-400 font-bold">{str}</span>;
        if (str.includes('Deleted')) return <span className="text-rose-400 font-bold">{str}</span>;
        if (str.includes('Loaded')) return <span className="text-sky-400 font-bold">{str}</span>;
        return str;
    };

    if (foundAgent) {
        let parts = text.split(foundAgent.name);
        if (parts.length === 2) {
            return (
                <span>
                    {colorizeKeywords(parts[0])}
                    <span className={`font-bold ${agentColor} bg-slate-800/50 px-1 py-0.5 rounded shadow-sm mx-0.5`}>
                        {foundAgent.emoji} {foundAgent.name}
                    </span>
                    {colorizeKeywords(parts[1])}
                </span>
            );
        }
        parts = text.split(foundAgent.id);
        if (parts.length === 2) {
            return (
                <span>
                    {colorizeKeywords(parts[0])}
                    <span className={`font-bold ${agentColor} bg-slate-800/50 px-1 py-0.5 rounded shadow-sm mx-0.5`}>
                        {foundAgent.emoji} {foundAgent.id}
                    </span>
                    {colorizeKeywords(parts[1])}
                </span>
            );
        }
    }

    return colorizeKeywords(text);
}



export default function MessageBubble({ msg, onShowDetails }) {
    const isUser = msg.from === 'User' || msg.type === 'user';
    const isSystem = msg.type === 'system';
    const isOrch = msg.from === 'Orchestrator' || msg.type === 'orchestrator';

    if (isSystem) {
        return (
            <div className="flex flex-col text-[11px] text-slate-400 font-mono my-2 py-2 px-3 border-l-2 border-indigo-500/30 bg-indigo-500/5 rounded-r-md">
                <div className="font-bold text-slate-500 mb-1 flex items-center gap-2">
                    <span>SYSTEM</span>
                    <span className="text-slate-600">⋙</span>
                    <span>LOG</span>
                </div>
                <div className={msg.error ? "text-red-400" : "text-slate-300"}>{formatSystemLog(msg.content)}</div>
            </div>
        )
    }

    return (
        <div className={`flex flex-col p-3.5 rounded-2xl border backdrop-blur-sm transition-all duration-300 relative ${isUser
            ? 'bg-gradient-to-br from-[#1e293b]/70 to-[#0f172a]/70 border-[#334155] ml-8 rounded-tr-sm shadow-md'
            : 'bg-gradient-to-b from-[#162135]/90 to-[#0f172a]/90 border-indigo-500/20 mr-8 rounded-tl-sm shadow-lg shadow-indigo-900/10'
            }`}>

            {/* JSON Info Icon */}
            {(msg.jsonPayload || (msg.logs && msg.logs.length > 0)) && (
                <div className="absolute -top-2 -left-2 group/info z-10 w-5 h-5">
                    <button
                        onClick={() => onShowDetails?.({ payload: msg.jsonPayload, logs: msg.logs })}
                        className="w-full h-full rounded-full bg-[#1e293b] border border-sky-500/50 flex items-center justify-center text-sky-400 hover:bg-sky-500 hover:text-white transition-all shadow-md cursor-pointer"
                    >
                        <span className="text-[10px] font-bold italic font-serif">i</span>
                    </button>
                    <div className="absolute left-6 top-0 hidden group-hover/info:block bg-[#0f172a] text-[10px] text-slate-200 px-2 py-1 rounded shadow-xl whitespace-nowrap z-20 border border-slate-700 pointer-events-none">
                        View Details
                    </div>
                </div>
            )}

            <div className="flex items-center text-[10px] tracking-[0.1em] font-bold mb-2.5 uppercase text-slate-400 flex-wrap shrink-0">
                <span className={isUser ? 'text-sky-400' : 'text-indigo-400'}>{msg.from}</span>
                <span className="mx-2 text-slate-600">⋙</span>
                {isOrch ? formatAgents(msg.to) : <span className="text-slate-300">{msg.to}</span>}
            </div>

            <div className={`text-[15px] leading-relaxed whitespace-pre-wrap ${msg.error ? 'text-red-400' : 'text-[#f1f5f9]'}`}>
                {msg.content}
                {msg.streaming && <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-indigo-400/80 animate-pulse">▍</span>}
            </div>

            {/* Media Embeds */}
            {!msg.streaming && (
                <div className="mt-3 space-y-3">
                    {/* Check top-level media from extractMedia */}
                    {msg.youtubeId && (
                        <div className="rounded-xl overflow-hidden border border-white/10 shadow-2xl aspect-video w-full bg-black">
                            <iframe
                                width="100%"
                                height="100%"
                                src={`https://www.youtube.com/embed/${msg.youtubeId}?autoplay=1`}
                                title="YouTube video player"
                                frameBorder="0"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowFullScreen
                            ></iframe>
                        </div>
                    )}

                    {msg.images?.map((img, i) => (
                        <img
                            key={i}
                            src={img}
                            alt=""
                            style={{ width: 220 }}
                        />
                    ))}

                    {/* Check jsonPayload for legacy support or alternative format */}
                    {msg.jsonPayload && Array.isArray(msg.jsonPayload) && msg.jsonPayload.map((resp, ridx) => (
                        resp.urls && resp.urls.map((url, uidx) => {
                            const ytId = getYoutubeId(url);
                            if (ytId && ytId !== msg.youtubeId) {
                                return (
                                    <div key={`json-${ridx}-${uidx}`} className="rounded-xl overflow-hidden border border-white/10 shadow-2xl aspect-video w-full bg-black">
                                        <iframe
                                            width="100%"
                                            height="100%"
                                            src={`https://www.youtube.com/embed/${ytId}?autoplay=0`}
                                            title="YouTube video player"
                                            frameBorder="0"
                                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                            allowFullScreen
                                        ></iframe>
                                    </div>
                                );
                            }
                            return null;
                        })
                    ))}
                </div>
            )}
        </div>
    )
}
