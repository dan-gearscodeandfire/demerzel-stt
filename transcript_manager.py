import difflib
from collections import deque


class TranscriptManager:
    """
    Manages transcription text with intelligent deduplication.
    Stores coherent speech snapshots, replacing extensions of previous
    transcript and appending new distinct thoughts.
    """

    def __init__(self, max_history=50):
        self.current_text = ""
        self.final_transcripts = deque(maxlen=max_history)
        self.accumulated_text = ""

    def update_current_text(self, text):
        if not text:
            return False
        if text != self.current_text:
            self.current_text = text
            return True
        return False

    def force_finalize_text(self):
        """Force finalize current text into accumulated buffer (long utterance handling)."""
        if not self.current_text or len(self.current_text.strip()) < 3:
            return False
        new_text = self.current_text.strip()
        new_text = new_text.replace("...", "").replace("\u2026", "")
        new_text = " ".join(new_text.split())
        if self.accumulated_text:
            self.accumulated_text += " " + new_text
        else:
            self.accumulated_text = new_text
        self.current_text = ""
        return True

    def finalize_text(self):
        """Finalize on pause detection with intelligent deduplication."""
        text_to_finalize = ""
        if self.accumulated_text:
            if self.current_text:
                text_to_finalize = self.accumulated_text + self.current_text.strip()
            else:
                text_to_finalize = self.accumulated_text
        elif self.current_text:
            text_to_finalize = self.current_text.strip()

        if not text_to_finalize or len(text_to_finalize) < 3:
            self.current_text = ""
            self.accumulated_text = ""
            return False

        new_text = text_to_finalize.replace("...", "").replace("\u2026", "")
        new_text = " ".join(new_text.split())
        new_text_lower = new_text.lower()

        if not self.final_transcripts:
            self.final_transcripts.append(new_text)
            self.current_text = ""
            self.accumulated_text = ""
            return True

        last_text = self.final_transcripts[-1]
        last_text_lower = last_text.lower()

        # Skip if new text is subset of last
        if new_text_lower in last_text_lower and len(new_text) < len(last_text):
            self.current_text = ""
            self.accumulated_text = ""
            return False

        # Check similarity for corrections/extensions
        ratio = difflib.SequenceMatcher(None, last_text_lower, new_text_lower).ratio()
        if ratio > 0.7:
            if len(new_text) >= len(last_text):
                self.final_transcripts[-1] = new_text
            self.current_text = ""
            self.accumulated_text = ""
            return True

        # Clear extension
        if last_text_lower in new_text_lower:
            self.final_transcripts[-1] = new_text
            self.current_text = ""
            self.accumulated_text = ""
            return True

        # New thought
        self.final_transcripts.append(new_text)
        self.current_text = ""
        self.accumulated_text = ""
        return True

    def get_current_text(self):
        return self.current_text

    def get_final_transcripts(self):
        return list(self.final_transcripts)

    def clear_all(self):
        self.current_text = ""
        self.accumulated_text = ""
        self.final_transcripts.clear()
