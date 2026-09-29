DESCRIPTION = """This agent handles infotainment and entertainment requests such as playing music, radio, podcasts, audiobooks, or telling jokes and funny stories.
It selects audio or online media from the infotainment system or connected services like radio, playlists, or YouTube, and returns content ready for playback.
Examples include play music, play a song, put on a playlist, tell a joke, play radio news, play a podcast, find a track on YouTube, or play background music."""

def build_orchestrator_description() -> str:
    return f"""DESCRIPTION: {DESCRIPTION}"""