import re
from collections import Counter

# шаблоны для каждой сущности, по которым я ищу упоминания в транскриптах
ENTITY_GROUPS = {
    "дистрибутив": {
        "Ubuntu": r"ubuntu|убунт\w*",
        "Linux Mint": r"linux mint|\bmint\b|\bминт\w*",
        "Debian": r"debian|дебиан\w*|дебьян\w*",
        "Fedora": r"fedora|федор[аеуыой]\b",
        "Arch Linux": r"\barch\b|\bарч\w*",
        "Kali Linux": r"\bkali\b|\bкали\b|\bкали линукс",
        "Astra Linux": r"\bastra\b|\bастр[аеуыой]\b|\bастралинукс\w*",
        "РЕД ОС": r"red ?os|\bред ?ос\w*|\bредос\w*",
        "CachyOS": r"cachy ?os|cachy|\bкэши ?ос\w*|\bкеши ?ос\w*",
        "Bazzite": r"bazzite|\bбаз+ит\w*",
        "Manjaro": r"manjaro|манджаро",
        "openSUSE": r"open ?suse|\bсузе\b|\bсюзе\b",
        "Gentoo": r"gentoo|генту",
        "SteamOS": r"steam ?os|стим ?ос",
        "Windows": r"windows|\bвинд\w*|\bвиндоус\w*",
        "macOS": r"mac ?os|\bмакос\w*|\bмак ос\b",
    },
    "окружение": {
        "GNOME": r"gnome|\bгном\w*",
        "KDE Plasma": r"\bkde\b|plasma|\bплазм\w*|\bкеды\b|\bкдэ\b",
        "Cinnamon": r"cinnamon|\bсин+амон\w*|\bцин+амон\w*",
        "XFCE": r"xfce|\bиксфце\b",
        "Fly": r"\bfly\b|\bфлай\b",
    },
    "пакеты": {
        "apt": r"\bapt\b|\bапт\b",
        "dnf": r"\bdnf\b|\bдэнэф\b",
        "pacman": r"pacman|\bпакман\w*",
        "AUR": r"\baur\b|\bаур\b",
        "Flatpak": r"flatpak|флэтпак\w*|флатпак\w*",
        "Snap": r"\bsnap\b|\bснап\w*|\bснэп\w*",
    },
    "железо": {
        "NVIDIA": r"nvidia|\bн?видиа\b|\bнвиди\w*|\bэнвиди\w*",
        "AMD": r"\bamd\b|\bамд\b|\bэйэмди\b",
        "Intel": r"\bintel\b|\bинтел\w*",
    },
    "программы": {
        "Steam": r"\bsteam\b|\bстим\b|\bстима\b|\bстиме\b",
        "Proton/Wine": r"proton|\bwine\b|\bпротон\w*|\bвайн\w*",
        "Docker": r"docker|\bдокер\w*",
        "Терминал": r"terminal|\bтерминал\w*|\bконсол\w*",
        "Драйверы": r"driver|\bдрайвер\w*",
    },
}

_COMPILED = [
    (group, name, re.compile(pattern, re.IGNORECASE))
    for group, items in ENTITY_GROUPS.items()
    for name, pattern in items.items()
]


def find_entities(text):
    """Возвращаю пары (группа, сущность) для всех упоминаний в тексте."""
    text = (text or "").lower().replace("ё", "е")
    found = []
    for group, name, regex in _COMPILED:
        found.extend((group, name) for _ in regex.finditer(text))
    return found


def count_entities(segments):
    counter = Counter()
    for seg in segments:
        counter.update(name for _, name in find_entities(seg.get("text", "")))
    return counter


def entity_group(name):
    for group, items in ENTITY_GROUPS.items():
        if name in items:
            return group
    return ""
