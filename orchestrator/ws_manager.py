from fastapi import WebSocket
from typing import Dict
 
class WebSocketManager:
 
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
 
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[session_id] = websocket
 
    def disconnect(self, session_id: str):
        if session_id in self.connections:
            del self.connections[session_id]
 
    async def send(self, session_id: str, message: dict):
        ws = self.connections.get(session_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception:
                pass

    async def broadcast(self, message: dict):
        for ws in list(self.connections.values()):
            try:
                await ws.send_json(message)
            except Exception:
                pass
 