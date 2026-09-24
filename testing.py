import re
import unicodedata
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend


def clean_and_normalize_nepali(text: str) -> str:
    """Normalizes Devanagari Unicode characters, strips non-Nepali symbols/punctuation,

    and standardizes whitespaces.
    """
    # 1. Unicode NFKC Normalization (standardizes combined glyphs/diacritics)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove URLs, email addresses, and metadata tags
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Keep only Devanagari characters, Dev spaces, and Devanagari numbers (\u0900-\u097F)
    # Replaces everything else (Latin letters, emojis, punctuation like commas, quotes, etc.) with a space
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)

    # 4. Remove standard Devanagari punctuation (e.g., Purna Viram '।', Double Danda '॥')
    text = re.sub(r"[।॥]", "", text)

    # 5. Normalize multiple spaces into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_nepali_phonemes(
    text: str, language: str = "ne", strip_stress: bool = True
) -> str:
    """Converts normalized Devanagari text into standardized IPA phonemes."""
    cleaned_text = clean_and_normalize_nepali(text)

    if not cleaned_text:
        return ""

    # Convert text to phonemes using eSpeak-ng backend
    phonemes = phonemize(
        cleaned_text,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=not strip_stress,
        njobs=1,
    )

    # Clean up excess internal phonetic spaces
    phonemes = re.sub(r"\s+", " ", phonemes).strip()
    return phonemes


# --- Example Usage ---
if __name__ == "__main__":
    raw_input = "नमस्ते! मेरो नाम... 'राम' हो। (welcome to Nepal, 2026!)"

    cleaned_text = clean_and_normalize_nepali(raw_input)
    phonemes = get_nepali_phonemes(raw_input)

    print(f"Raw Input:      {raw_input}")
    print(f"Cleaned Text:   {cleaned_text}")
    print(f"Phonemes (IPA): {phonemes}")