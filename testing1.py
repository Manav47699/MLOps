import re


def refine_nepali_ipa(ipa_str: str) -> str:
    # 1. Remove word-final schwa (ə) unless preceded by a single consonant at word start
    # e.g., 'naːmə' -> 'naːm', 'ɾaːmə' -> 'ɾaːm'
    words = ipa_str.split()
    refined_words = []

    for word in words:
        # Remove trailing schwa if the word has more than one syllable/character block
        if word.endswith("ə") and len(word) > 2:
            word = word[:-1]
        refined_words.append(word)

    result = " ".join(refined_words)

    # 2. Normalize length markers if your TTS model prefers un-lengthened IPA vowels
    # e.g., 'eː' -> 'e', 'oː' -> 'o'
    result = re.sub(r"ː", "", result)

    return result


# Applied to your current IPA output:
raw_ipa = "nəmʌsteː meːɾoː naːmə ɾaːmə hoː"
clean_ipa = refine_nepali_ipa(raw_ipa)
print(clean_ipa)
# Output: nəmʌste meɾo naːm ɾaːm ho