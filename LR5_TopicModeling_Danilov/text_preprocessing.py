"""
Предобработка русскоязычных текстов для ЛР5.

Конвейер: нижний регистр -> токенизация -> удаление стоп-слов ->
лемматизация (pymorphy3) или, если pymorphy3 не установлен, стемминг
Snowball для русского языка. Для стемов дополнительно запоминается самая
частая исходная словоформа, чтобы ключевые слова тем оставались читаемыми.
"""

import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-zа-яё][a-zа-яё\-]*[a-zа-яё]|[a-zа-яё]")

STOPWORDS_RU = set("""
а без более больше будет будут бы был была были было быть в вам вас весь во вот все всего всех вы
где да даже для до его ее её если есть еще ещё же за здесь и из или им их к как ко когда кто ли
либо мне может мы на надо наш не него нее неё нет ни них но ну о об однако он она они оно от
очень по под при с со так также такой там те тем то того тоже той только том ты у уже хотя
чего чей чем что чтобы чье чья эта эти это этого этой этом этот я
который которая которое которые которых которым которой котором
свой своя свое свои своих своим можно нужно нужен нужна также поэтому потому чтоб
сам сама сами самый самая самое себя себе сейчас тут теперь тогда этим этих всё всем всю
весь вся всё каждый каждая другой другие один одна одно два три раз ещё лишь просто
был быть есть стал стала стали стать будет бывает являться является являются
год года лет году день дня дни раз время рамках также например т.е тд др
очень хорошо хороший хорошая отлично плохо пока всегда никогда почти совсем много мало
мой моя мое мои наш наша наше наши ваш ваша ваше ваши его её их
какой какая какое какие сколько почему зачем куда откуда
через после перед между около вокруг против вместо ради вне среди из-за из-под
""".split())

STOPWORDS_EN = set("""
a an the and or but if of to in on for with at by from as is are was were be been being it its
this that these those i you he she we they them his her our your their not no yes do does did
so than then there here what which who whom how why when where all any can will just about into
""".split())

DOMAIN_STOPWORDS = set("""
шт см мм гб тб мб ггц мгц руб ₽ мл кг гр
habr хабр статья статье статью статьи пост
сделал сделала сделали сделать часть новый новая новое новые одной одного одном
про меня мне нам нас вам ему ей ним ней них нем нём образом месте пример порядка первый первая внутри
блог компания компании компаний спасибо рекомендую
товар товара товаров товаре заказ заказа заказала заказал пришел пришёл пришла пришло
""".split())

STOPWORDS = STOPWORDS_RU | STOPWORDS_EN | DOMAIN_STOPWORDS


class SnowballRussian:
    """Реализация алгоритма Snowball (Porter) для русского языка."""

    VOWELS = set("аеиоуыэюя")

    PERFECTIVE_1 = ("вшись", "вши", "в")
    PERFECTIVE_2 = ("ившись", "ывшись", "ивши", "ывши", "ив", "ыв")
    REFLEXIVE = ("ся", "сь")
    ADJECTIVE = ("ими", "ыми", "его", "ого", "ему", "ому", "ее", "ие", "ые", "ое", "ей", "ий",
                 "ый", "ой", "ем", "им", "ым", "ом", "их", "ых", "ую", "юю", "ая", "яя", "ою", "ею")
    PARTICIPLE_1 = ("ем", "нн", "вш", "ющ", "щ")
    PARTICIPLE_2 = ("ивш", "ывш", "ующ")
    VERB_1 = ("ете", "йте", "ешь", "нно", "ла", "на", "ли", "ем", "ло", "но", "ет", "ют", "ны", "ть", "й", "л", "н")
    VERB_2 = ("ейте", "уйте", "ила", "ыла", "ена", "ите", "или", "ыли", "ило", "ыло", "ено", "ует",
              "уют", "ены", "ить", "ыть", "ишь", "ей", "уй", "ил", "ыл", "им", "ым", "ен", "ят", "ит", "ыт", "ую", "ю")
    NOUN = ("иями", "ями", "ами", "ией", "иям", "ием", "иях", "ев", "ов", "ие", "ье", "еи", "ии", "ей",
            "ой", "ий", "ям", "ем", "ам", "ом", "ах", "ях", "ию", "ью", "ия", "ья", "а", "е", "и", "й",
            "о", "у", "ы", "ь", "ю", "я")
    SUPERLATIVE = ("ейше", "ейш")
    DERIVATIONAL = ("ость", "ост")

    def __init__(self):
        self.cache = {}

    def _regions(self, word):
        rv = len(word)
        for i, ch in enumerate(word):
            if ch in self.VOWELS:
                rv = i + 1
                break
        r1 = len(word)
        for i in range(1, len(word)):
            if word[i] not in self.VOWELS and word[i - 1] in self.VOWELS:
                r1 = i + 1
                break
        r2 = len(word)
        for i in range(r1 + 1, len(word)):
            if word[i] not in self.VOWELS and word[i - 1] in self.VOWELS:
                r2 = i + 1
                break
        return rv, r2

    @staticmethod
    def _strip(rv_part, endings, preceded=None):
        for ending in sorted(endings, key=len, reverse=True):
            if rv_part.endswith(ending):
                head = rv_part[: -len(ending)]
                if preceded is None:
                    return head, True
                if head and head[-1] in preceded:
                    return head, True
        return rv_part, False

    def stem(self, word):
        if word in self.cache:
            return self.cache[word]

        original = word
        word = word.replace("ё", "е")
        rv, r2 = self._regions(word)
        prefix, rv_part = word[:rv], word[rv:]

        rv_part, done = self._strip(rv_part, self.PERFECTIVE_1, preceded="ая")
        if not done:
            rv_part, done = self._strip(rv_part, self.PERFECTIVE_2)
        if not done:
            rv_part, _ = self._strip(rv_part, self.REFLEXIVE)
            rv_part, done = self._strip(rv_part, self.ADJECTIVE)
            if done:
                rv_part, found = self._strip(rv_part, self.PARTICIPLE_1, preceded="ая")
                if not found:
                    rv_part, _ = self._strip(rv_part, self.PARTICIPLE_2)
            else:
                rv_part, done = self._strip(rv_part, self.VERB_1, preceded="ая")
                if not done:
                    rv_part, done = self._strip(rv_part, self.VERB_2)
                if not done:
                    rv_part, _ = self._strip(rv_part, self.NOUN)

        if rv_part.endswith("и"):
            rv_part = rv_part[:-1]

        word = prefix + rv_part
        r2_part = word[r2:] if r2 < len(word) else ""
        for ending in self.DERIVATIONAL:
            if r2_part.endswith(ending):
                word = word[: -len(ending)]
                break

        rv_part = word[rv:]
        if rv_part.endswith("нн"):
            rv_part = rv_part[:-1]
        else:
            stripped, found = self._strip(rv_part, self.SUPERLATIVE)
            if found:
                rv_part = stripped[:-1] if stripped.endswith("нн") else stripped
            elif rv_part.endswith("ь"):
                rv_part = rv_part[:-1]
        word = prefix + rv_part

        self.cache[original] = word
        return word


class Normalizer:
    """Лемматизатор pymorphy3 или стеммер Snowball (запасной вариант)."""

    def __init__(self, mode="auto"):
        self.morph = None
        self.mode = "stem"
        if mode in ("auto", "lemma"):
            try:
                import pymorphy3
                self.morph = pymorphy3.MorphAnalyzer()
                self.mode = "lemma"
            except ImportError:
                if mode == "lemma":
                    raise
        self.stemmer = SnowballRussian()
        self.cache = {}
        self.surface = defaultdict(Counter)

    def __call__(self, token):
        if token in self.cache:
            norm = self.cache[token]
        else:
            if self.mode == "lemma":
                norm = self.morph.parse(token)[0].normal_form
            elif re.search("[а-яё]", token):
                norm = self.stemmer.stem(token)
            else:
                norm = token
            self.cache[token] = norm
        self.surface[norm][token] += 1
        return norm

    def display(self, norm):
        if self.mode == "lemma" or norm not in self.surface:
            return norm
        return self.surface[norm].most_common(1)[0][0]


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower().replace("ё", "е"))


def preprocess(texts, normalizer, min_len=3, extra_stopwords=()):
    stop = STOPWORDS | set(extra_stopwords)
    result = []
    for text in texts:
        tokens = []
        for token in tokenize(text):
            if len(token) < min_len or token in stop or token.isdigit():
                continue
            norm = normalizer(token)
            if norm in stop or len(norm) < 2:
                continue
            tokens.append(norm)
        result.append(tokens)
    return result
