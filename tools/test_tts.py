from __future__ import annotations

import time

from navai.audio.tts import VoiceAlertEngine


def main() -> int:
    voice = VoiceAlertEngine(cooldown_s=0)
    voice.say("test", "NAVAI voice test. If you can hear this, text to speech is working.", force=True)
    time.sleep(8)
    voice.say("assistant", "This is an assistant answer test.", force=True)
    time.sleep(8)
    voice.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
