from emoji.unicode_codes import UNICODE_EMOJI


def get_unicode_emoji_from_text(text):
    emojis = []
    for char in text:
        if char in UNICODE_EMOJI.keys():
            emojis.append(char)
    return emojis
