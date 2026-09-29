import base64
import json
import os
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse

import numpy as np

from stt import ElevenLabsRealtimeTranscriptionBackend, STT_SAMPLE_RATE


class FakeWebSocket:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(json.loads(payload))

    def recv(self):
        if not self.responses:
            return None
        return json.dumps(self.responses.pop(0))

    def close(self):
        self.closed = True


class ElevenLabsRealtimeTranscriptionBackendTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            os.environ,
            {
                "ELEVENLABS_API_KEY": "test-key",
                "ELEVENLABS_STT_MODEL": "scribe_v2_realtime",
                "ELEVENLABS_STT_URL": "wss://api.elevenlabs.io/v1/speech-to-text/realtime",
                "ELEVENLABS_STT_TIMEOUT": "20",
                "ELEVENLABS_STT_INCLUDE_TIMESTAMPS": "false",
                "ELEVENLABS_STT_NO_VERBATIM": "false",
                "ELEVENLABS_STT_COMMIT_STRATEGY": "manual",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def _backend(self, responses, language="en", sockets=None):
        sockets = sockets if sockets is not None else []

        def factory(url, header=None, timeout=None):
            ws = FakeWebSocket(responses)
            ws.url = url
            ws.header = header
            ws.timeout = timeout
            sockets.append(ws)
            return ws

        return ElevenLabsRealtimeTranscriptionBackend(
            device_sample_rate=STT_SAMPLE_RATE,
            language=language,
            websocket_factory=factory,
        )

    def test_missing_api_key_raises_clear_error(self):
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "ELEVENLABS_API_KEY"):
                ElevenLabsRealtimeTranscriptionBackend(STT_SAMPLE_RATE)

    def test_sends_pcm16_base64_chunk_and_parses_committed_transcript(self):
        sockets = []
        backend = self._backend(
            [
                {"message_type": "session_started", "session_id": "abc"},
                {"message_type": "partial_transcript", "text": "hello"},
                {"message_type": "committed_transcript", "text": "hello world"},
            ],
            sockets=sockets,
        )

        audio = np.array([0, 1000, -1000, 2000], dtype=np.int16)
        result = backend.transcribe(audio)

        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["language"], "en")
        self.assertTrue(sockets[0].closed)
        self.assertEqual(sockets[0].header, ["xi-api-key: test-key"])
        self.assertEqual(sockets[0].timeout, 20.0)

        query = parse_qs(urlparse(sockets[0].url).query)
        self.assertEqual(query["model_id"], ["scribe_v2_realtime"])
        self.assertEqual(query["audio_format"], ["pcm_16000"])
        self.assertEqual(query["language_code"], ["en"])
        self.assertEqual(query["commit_strategy"], ["manual"])

        sent = sockets[0].sent[0]
        self.assertEqual(sent["message_type"], "input_audio_chunk")
        self.assertEqual(sent["sample_rate"], STT_SAMPLE_RATE)
        self.assertIs(sent["commit"], True)
        decoded = base64.b64decode(sent["audio_base_64"])
        self.assertEqual(decoded, audio.tobytes())

    def test_parses_committed_transcript_with_timestamps_language(self):
        backend = self._backend(
            [
                {
                    "message_type": "committed_transcript_with_timestamps",
                    "text": "xin chao",
                    "language_code": "vi",
                    "words": [],
                },
            ],
            language=None,
        )

        result = backend.transcribe(np.array([0, 1000], dtype=np.int16))

        self.assertEqual(result["text"], "xin chao")
        self.assertEqual(result["language"], "vi")
        self.assertEqual(result["generated_language"], "vi")

    def test_error_event_raises_runtime_error(self):
        backend = self._backend(
            [
                {
                    "message_type": "scribe_rate_limited_error",
                    "message": "too many requests",
                },
            ],
        )

        with self.assertRaisesRegex(RuntimeError, "scribe_rate_limited_error"):
            backend.transcribe(np.array([0, 1000], dtype=np.int16))


if __name__ == "__main__":
    unittest.main()
