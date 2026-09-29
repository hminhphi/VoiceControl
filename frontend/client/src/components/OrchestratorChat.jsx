import React, { useEffect, useRef, useState } from "react"
import { CATALOG } from "../App"
import MessageBubble from "./MessageBubble"
import { extractMedia, getYoutubeId } from "../utils/mediaUtils"
import { AGENT_COLORS } from "../utils/chatUtils"

const ORCH_BASE_RAW = import.meta.env.VITE_ORCHESTRATOR_URL || "http://localhost:8000"
const ORCH_BASE = ORCH_BASE_RAW.replace(/\/+$/, "")
const WS_BASE = ORCH_BASE.replace(/^http/, "ws")

const API_URL = `${ORCH_BASE}/v1/orchestrator/message`
const WS_URL = `${WS_BASE}/v1/orchestrator/ws`
const FRONTEND_SESSION_ID = "frontend"

function uuid() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID()
    }
    const random = Math.random().toString(36).slice(2)
    const timestamp = Date.now().toString(36)
    return `${timestamp}-${random}`
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

    // Helper to colorize keywords in a segment
    const colorizeKeywords = (str) => {
        if (str.includes('Added')) return <span className="text-emerald-400 font-bold">{str}</span>;
        if (str.includes('Deleted')) return <span className="text-rose-400 font-bold">{str}</span>;
        if (str.includes('Loaded')) return <span className="text-sky-400 font-bold">{str}</span>;
        return str;
    };

    if (foundAgent) {
        // Try naming match
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
        // Try ID match
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

const highlightJson = (obj) => {
    const json = JSON.stringify(obj, null, 2);
    return json.split('\n').map((line, i) => {
        // Simple regex highlighting
        const keyMatch = line.match(/^(\s*)"([^"]+)"(?=:)/);
        const stringMatch = line.match(/: "([^"]*)"/);
        const numberMatch = line.match(/: ([\d.]+)/);
        const boolMatch = line.match(/: (true|false|null)/);

        if (keyMatch) {
            return (
                <div key={i} className="whitespace-pre">
                    {keyMatch[1]}<span className="text-sky-400">"{keyMatch[2]}"</span>:
                    {line.substring(keyMatch[0].length + 1)}
                </div>
            );
        }

        // Colorize values if no key match on this line (or parts of it)
        let coloredLine = line;
        if (stringMatch) return <div key={i} className="whitespace-pre text-emerald-300">{line}</div>;
        if (numberMatch) return <div key={i} className="whitespace-pre text-amber-300">{line}</div>;
        if (boolMatch) return <div key={i} className="whitespace-pre text-pink-400">{line}</div>;

        return <div key={i} className="whitespace-pre text-slate-400">{line}</div>;
    });
}



export default function OrchestratorChat({ setActiveAgents, log = [], setIsDone }) {

    const sessionId = useRef(FRONTEND_SESSION_ID)
    const wsRef = useRef(null)
    const chatContainerRef = useRef(null)

    const [message, setMessage] = useState("")
    const [chatHistory, setChatHistory] = useState([])
    const [selectedJson, setSelectedJson] = useState(null)
    const currentMsgId = useRef(null)
    const lastLogIndex = useRef(0)
    const voiceSessionToMsgId = useRef({})
    const voiceSessionToUserMsgId = useRef({})
    const voiceSessionToAgentIds = useRef({})

    // ----------------------------
    // SYNC SYSTEM LOGS
    // ----------------------------
    useEffect(() => {
        if (log && log.length > lastLogIndex.current) {
            const newLogs = log.slice(lastLogIndex.current);
            setChatHistory(prev => [
                ...prev,
                ...newLogs.map((l, i) => ({
                    id: `sys-${lastLogIndex.current + i}`,
                    from: 'System',
                    to: 'Log',
                    content: l,
                    type: 'system'
                }))
            ]);
            lastLogIndex.current = log.length;
        }
    }, [log])

    // SCROLL TO BOTTOM
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory])

    // ---------------- WS ----------------

    useEffect(() => {

        const ws = new WebSocket(
            `${WS_URL}/${sessionId.current}`
        )

        ws.onmessage = (event) => {

            const data = JSON.parse(event.data)
            const isVoice = data.session_id && String(data.session_id).startsWith("voice-")

            function ensureVoiceBubble(sid) {
                if (voiceSessionToMsgId.current[sid]) return voiceSessionToMsgId.current[sid]
                const assistantMsgId = uuid()
                voiceSessionToMsgId.current[sid] = assistantMsgId
                setIsDone?.(false)
                return assistantMsgId
            }

            if (data.type === "user_message" && data.session_id && String(data.session_id).startsWith("voice-")) {
                const sid = data.session_id
                if (data.agent_ids) voiceSessionToAgentIds.current[sid] = data.agent_ids
                if (voiceSessionToUserMsgId.current[sid]) return
                const userMsgId = uuid()
                voiceSessionToUserMsgId.current[sid] = userMsgId
                setChatHistory(prev => [
                    ...prev,
                    { id: userMsgId, from: "User", to: "Orchestrator", content: data.content || "" }
                ])
                return
            }

            const resolvedId = isVoice
                ? (voiceSessionToMsgId.current[data.session_id] || ensureVoiceBubble(data.session_id))
                : currentMsgId.current
            if (!resolvedId) return

            const hasUserBubble = isVoice && voiceSessionToUserMsgId.current[data.session_id]
            const voiceAgentIds = isVoice && data.session_id ? (voiceSessionToAgentIds.current[data.session_id] || []) : []

            // ---------- DATA ----------

            if (data.type === "data") {

                const media = extractMedia(data.content || {})

                setChatHistory(prev => {
                    const hasMsg = prev.some(m => m.id === resolvedId)
                    if (isVoice && !hasMsg) {
                        const newMessages = hasUserBubble
                            ? [{ id: resolvedId, from: "Orchestrator", to: voiceAgentIds, content: "", streaming: true, ...media }]
                            : [
                                { id: uuid(), from: "User", to: "Orchestrator", content: "(Voice)" },
                                { id: resolvedId, from: "Orchestrator", to: voiceAgentIds, content: "", streaming: true, ...media }
                            ]
                        return [...prev, ...newMessages]
                    }
                    return prev.map(m => m.id === resolvedId ? { ...m, ...media } : m)
                })

                return
            }

            // ---------- TOKEN ----------

            if (data.type === "token") {

                setChatHistory(prev => {
                    const hasMsg = prev.some(m => m.id === resolvedId)
                    if (isVoice && !hasMsg) {
                        const newMessages = hasUserBubble
                            ? [{ id: resolvedId, from: "Orchestrator", to: voiceAgentIds, content: data.content || "", streaming: true }]
                            : [
                                { id: uuid(), from: "User", to: "Orchestrator", content: "(Voice)" },
                                { id: resolvedId, from: "Orchestrator", to: voiceAgentIds, content: data.content || "", streaming: true }
                            ]
                        return [...prev, ...newMessages]
                    }
                    return prev.map(m =>
                        m.id === resolvedId ? { ...m, content: m.content + (data.content || "") } : m
                    )
                })

                return
            }

            // ---------- DONE ----------

            if (data.type === "done") {

                setChatHistory(prev => {
                    const hasMsg = prev.some(m => m.id === resolvedId)
                    if (isVoice && !hasMsg) {
                        const newMessages = hasUserBubble
                            ? [{ id: resolvedId, from: "Orchestrator", to: voiceAgentIds, content: data.message || "", streaming: false }]
                            : [
                                { id: uuid(), from: "User", to: "Orchestrator", content: "(Voice)" },
                                { id: resolvedId, from: "Orchestrator", to: voiceAgentIds, content: data.message || "", streaming: false }
                            ]
                        return [...prev, ...newMessages]
                    }
                    return prev.map(m =>
                        m.id === resolvedId ? { ...m, streaming: false, content: data.message || m.content } : m
                    )
                })

                setIsDone?.(true)
                setActiveAgents?.([])

                return
            }

            // ---------- ERROR ----------

            if (data.type === "error") {

                setChatHistory(prev => {
                    const hasMsg = prev.some(m => m.id === resolvedId)
                    if (isVoice && !hasMsg) {
                        const newMessages = hasUserBubble
                            ? [{ id: resolvedId, from: "Orchestrator", to: voiceAgentIds, content: "ERROR: " + (data.message || ""), streaming: false, error: true }]
                            : [
                                { id: uuid(), from: "User", to: "Orchestrator", content: "(Voice)" },
                                { id: resolvedId, from: "Orchestrator", to: voiceAgentIds, content: "ERROR: " + (data.message || ""), streaming: false, error: true }
                            ]
                        return [...prev, ...newMessages]
                    }
                    return prev.map(m =>
                        m.id === resolvedId ? { ...m, streaming: false, content: "ERROR: " + (data.message || ""), error: true } : m
                    )
                })

                setIsDone?.(true)
                setActiveAgents?.([])

                return
            }

            if (data.type === "backend_log") {
                setChatHistory(prev => prev.map(msg =>
                    msg.id === resolvedId
                        ? { ...msg, logs: [...(msg.logs || []), data.content] }
                        : msg
                ));
            }
        }

        wsRef.current = ws

        return () => ws.close()

    }, [])

    // ---------------- SEND ----------------

    const sendMessage = async () => {

        if (!message.trim()) return

        const text = message
        setMessage("")

        const payload = {
            message: text,
            session_id: sessionId.current
        }

        setChatHistory(prev => [

            ...prev,

            {
                id: uuid(),
                from: "User",
                to: "Orchestrator",
                content: text
            }

        ])

        setIsDone?.(false)

        const res = await fetch(
            API_URL,
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify(
                    payload
                )
            }
        )

        const data = await res.json()

        const nextId = uuid()

        currentMsgId.current = nextId

        setChatHistory(prev => [

            ...prev,

            {
                id: nextId,
                from: "Orchestrator",
                to: data.agent_ids,
                content: "",
                streaming: true
            }

        ])

        if (setActiveAgents) {
            setActiveAgents(data.agent_ids)
        }

    }

    // ---------------- UI ----------------

    return (

        <div className="flex flex-col h-full bg-[#0a0f1a]/80 backdrop-blur-md rounded-xl flex-1 p-0 text-[#cfe1ff] min-h-0 border border-[#1e293b] shadow-2xl relative overflow-hidden w-full">

            {/* Elegant Header */}
            <div className="flex items-center px-4 py-3 bg-[#0d1424] border-b border-[#1e293b]/80 shadow-md z-10 shrink-0">
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-[pulse_2s_ease-in-out_infinite] shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
                    <span className="text-sm font-bold tracking-widest text-slate-200 uppercase bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-sky-400">
                        Activity & Dialogue
                    </span>
                </div>
            </div>

            {/* Chat History */}
            <div ref={chatContainerRef} className="flex-1 overflow-y-auto min-h-0 p-4 space-y-4 custom-scrollbar bg-gradient-to-b from-transparent to-[#050810]/50 relative z-0">
                {chatHistory.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500/50 italic space-y-4">
                        <svg className="w-12 h-12 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0-4H3m18 0h-4m-7-7V3m0 0a7 7 0 01-7-7m7 7v4m0-4H3m18 0h-4" /></svg>
                        <span>Awaiting interactions...</span>
                    </div>
                )}

                {chatHistory.map((m) => (
                    <MessageBubble
                        key={m.id}
                        msg={m}
                        onShowDetails={setSelectedJson}
                    />
                ))}
            </div>


            {/* JSON/Logs Modal */}
            {selectedJson && (
                <div className="absolute inset-0 z-50 bg-black/60 backdrop-blur-sm flex flex-col p-4 animate-[fadeIn_0.2s_ease-out]">
                    <div className="bg-[#0f172a] border border-sky-500/30 rounded-xl shadow-2xl p-4 w-full h-full flex flex-col relative overflow-hidden">
                        <button
                            onClick={() => setSelectedJson(null)}
                            className="absolute top-3 right-3 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 w-6 h-6 rounded-full flex items-center justify-center transition-colors z-20"
                        >
                            ✕
                        </button>

                        <div className="flex gap-4 border-b border-[#1e293b] mb-4 pb-2 z-10">
                            <h3 className="text-sky-400 font-bold tracking-widest text-xs uppercase flex items-center gap-2">
                                <span className="italic font-serif bg-sky-500/20 w-5 h-5 rounded-full flex items-center justify-center border border-sky-500/50 text-sky-300">i</span>
                                Execution Details
                            </h3>
                        </div>

                        <div className="flex-1 overflow-y-auto space-y-6 custom-scrollbar pr-1">
                            {/* JSON Payload Section - MOVED UP */}
                            {selectedJson.payload && (
                                <div>
                                    <div className="text-[10px] font-bold text-slate-500 mb-2 uppercase tracking-widest pl-1">Data Payload</div>
                                    <div className="bg-[#050810] border border-[#1e293b] rounded-lg p-4 overflow-x-auto shadow-inner">
                                        <pre className="text-[11px] font-mono leading-relaxed">
                                            {highlightJson(selectedJson.payload)}
                                        </pre>
                                    </div>
                                </div>
                            )}

                            {/* Logs Section - MOVED DOWN */}
                            {selectedJson.logs && selectedJson.logs.length > 0 && (
                                <div>
                                    <div className="text-[10px] font-bold text-slate-500 mb-2 uppercase tracking-widest pl-1">Process Logs</div>
                                    <div className="bg-[#050810] border border-[#1e293b] rounded-lg p-0 overflow-hidden divide-y divide-[#1e293b]">
                                        {selectedJson.logs.map((l, i) => (
                                            <div key={i} className="text-[11px] font-mono leading-relaxed p-3 hover:bg-[#0c1426] transition-colors">
                                                {formatSystemLog(l)}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Input Area */}
            <div className="p-3 bg-[#0b1221] border-t border-[#1e293b] shrink-0 z-10 relative">
                <div className="relative flex group">
                    <input
                        value={message}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') sendMessage()
                        }}
                        onChange={(e) => setMessage(e.target.value)}
                        placeholder="Command the orchestrator..."
                        className="w-full bg-[#162032] border border-[#2d3748] rounded-xl pl-4 pr-12 py-3 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 text-white placeholder-slate-500"
                    />
                    <button
                        onClick={sendMessage}
                        className="absolute right-1.5 top-1.5 bottom-1.5 aspect-square flex items-center justify-center bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors group-focus-within:bg-indigo-500"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" /></svg>
                    </button>
                </div>
            </div>

        </div>

    )

}