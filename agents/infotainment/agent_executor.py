import json
import logging
import os

from typing_extensions import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from media_store import get_random_joke, get_song_by_title_or_id, get_song_titles, get_random_song
from transformers import pipeline
from youtube_search import YouTubeSearch
from utils import read_config  


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("a2a").setLevel(logging.WARNING)
logger = logging.getLogger("infotainment")

youtube_client = YouTubeSearch()

def _get_user_text(context: RequestContext) -> str:
    msg = getattr(context, "message", None)
    if not msg:
        request = getattr(context, "request", None)
        if request:
            params = getattr(request, "params", None)
            if params:
                msg = getattr(params, "message", None)
    if not msg:
        return ""
    parts = getattr(msg, "parts", None) or []
    for p in parts:
        part_root = getattr(p, "root", p)
        kind = getattr(part_root, "kind", None) or getattr(p, "type", None)
        if kind == "text":
            return getattr(part_root, "text", None) or getattr(p, "text", None) or ""
    return ""


class InfotainmentAgentExecutor(AgentExecutor):
    def __init__(self, base_url: str = "") -> None:
        self._base_url = base_url or os.environ.get("INFOTAINMENT_BASE_URL", "http://localhost:8004")
        
        config_path = "config.yaml"
        
        self.config = read_config(path=config_path).get('classifier', {})

        self.model = self.config.get("model", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
        self.candidate_labels = self.config.get("candidate_labels", ["play joke", "play music", "unknown label"])
        self.threshold = self.config.get("threshold", 0.4)
        self.multi_label = self.config.get("multi_label", False)
        
        self.load_classifier()

    def _find_snapshot_dir(self, cache_dir: str, model_id: str) -> str | None:
        direct_name = "models--" + model_id.replace("/", "--")
        prefixed_name = "models--sentence-transformers--" + model_id.split("/")[-1]
        for name in (direct_name, prefixed_name):
            for sub in ("hub", ""):
                base = os.path.join(cache_dir, sub, name) if sub else os.path.join(cache_dir, name)
                snapshots = os.path.join(base, "snapshots")
                if os.path.isdir(snapshots):
                    for child in os.listdir(snapshots):
                        full = os.path.join(snapshots, child)
                        if os.path.isdir(full) and os.listdir(full):
                            return full
        return None

    def load_classifier(self):
        cache_dir = os.path.abspath(os.environ.get("HF_HOME", "cache"))
        snapshot_dir = self._find_snapshot_dir(cache_dir, self.model)
        if snapshot_dir is not None:
            logger.info("Loading infotainment classifier from local snapshot: %s", snapshot_dir)
            model_to_load = snapshot_dir
        else:
            logger.info("Loading infotainment classifier from Hugging Face: %s (cache: %s)", self.model, cache_dir)
            model_to_load = self.model

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_to_load,
            device=0,
            model_kwargs={"cache_dir": cache_dir},
        )
        warmup_sequences = [
            "dummy sentence",
        ]
        _ = self.classifier(
            warmup_sequences,
            self.candidate_labels,
            multi_label=self.multi_label
        )

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        async def _out(success: bool, message: str, url: str | None = None, **meta) -> None:
            body = {
                "success": success,
                "message": message,
                "data": {
                    "url": url
                },
                "meta": meta or {}
            }

            await event_queue.enqueue_event(
                new_agent_text_message(json.dumps(body))
            )

        text = (_get_user_text(context) or "").strip()

        if not text:
            await _out(False, "Say 'play joke' or 'play music'.")
            return

        # classify
        results = self.classifier(
            [text],
            self.candidate_labels,
            multi_label=False
        )[0]

        logger.info(f"results: {results}")

        actions = []
        unknown_detected = False

        for label, score in zip(results["labels"], results["scores"]):
            if score >= self.threshold:
                if label == "unknown label":
                    unknown_detected = True
                else:
                    actions.append(label)

        if unknown_detected or not actions:
            await _out(False, "Say 'play joke' or 'play music'.")
            return

        top_action = actions[0]

        # ---------------- JOKE ----------------

        if top_action == "play joke":

            joke = get_random_joke()

            if not joke:
                await _out(False, "No jokes in library.")
                return

            joke_text = joke.get("text", "")

            await _out(
                True,
                f"here is the joke for you: {joke_text}",
            )
            return

        # ---------------- MUSIC ----------------

        if top_action == "play music":

            # try local song first
            song = get_song_by_title_or_id(text)

            if song:
                title = song.get("title") or song.get("id", "Track")
                song_path = song.get("path", "")

                url = f"{self._base_url.rstrip('/')}/static/songs/{song_path}"

                await _out(
                    True,
                    f"i'm playing {title}",
                    url=url,
                )
                return

            # fallback youtube
            yt_results = youtube_client.search(text)

            if yt_results:

                top_result = yt_results[0]

                title = top_result.get("title", "Unknown Title")
                url = top_result.get("url")

                await _out(
                    True,
                    f"youtube {title} is playing",
                    url=url,
                )
                return

            # no result
            titles = get_song_titles()

            if titles:
                msg = (
                    "That song isn't in the library. You can try: "
                    + ", ".join(titles)
                    + "."
                )
            else:
                msg = "No songs in library."

            await _out(True, msg)
            return

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


if __name__ == "__main__":
    import asyncio

    class MockEventQueue:
        async def enqueue_event(self, event):
            # event is an a2a Message object. In pydantic v2/latest, it might not have .root
            # We access parts directly.
            try:
                text = event.parts[0].text
                print(f"OUT: {text}")
            except (AttributeError, IndexError):
                # Fallback for structured parts
                try:
                    text = event.parts[0].root.text
                    print(f"OUT: {text}")
                except:
                    print(f"OUT: {event}")

    async def main():
        executor = InfotainmentAgentExecutor()
        queue = MockEventQueue()
        
        while True:
            try:
                user_input = input("\nInfotainment (joke/music) > ").strip()
                if user_input.lower() in [":q", "exit", "quit"]:
                    break
                if not user_input:
                    continue
                
                # Mock RequestContext
                class MockRequest:
                    def __init__(self, text):
                        # Construct a structure that _get_user_text can parse
                        # msg = getattr(context, "message", None)
                        # parts = getattr(msg, "parts", None)
                        # kind = getattr(part_root, "kind", None)
                        self.message = type('obj', (object,), {
                            'parts': [type('obj', (object,), {
                                'root': type('obj', (object,), {
                                    'kind': 'text',
                                    'text': text
                                })
                            })]
                        })
                
                context = MockRequest(user_input)
                await executor.execute(context, queue)
            except KeyboardInterrupt:
                break

    asyncio.run(main())

