import os
import re
from pathlib import Path
from urllib.parse import urlencode

import scrapy
from scrapy_playwright.page import PageMethod
from lab4_scraper.items import ProductItem

PROFILE_DIR = os.getenv("ECOMMERCE_PROFILE", str(Path.home() / "pw_profile"))
HEADLESS = os.getenv("ECOMMERCE_HEADLESS", "0") != "0"

BROWSER_OPTIONS = {
    "headless": HEADLESS,
    # Настоящий браузер вместо встроенного Chromium: msedge (есть в Windows 11) или chrome.
    "channel": os.getenv("ECOMMERCE_CHANNEL", "msedge"),
    "ignore_default_args": ["--enable-automation"],
    "args": ["--disable-blink-features=AutomationControlled"],
}

_TEXT_XPATH = (
    ".//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::noscript)"
    " and not(ancestor::template) and not(ancestor::textarea)]"
)


_JUNK_MARKERS = ("@marketfront", "apiary", "\"widgets\"", "window.", "function(", "=>")


def _is_junk_text(text):
    t = text.strip()
    if len(t) > 400:  # название/цена/бейдж никогда не бывают такими длинными
        return True
    if t[:1] in "{[" and ('":' in t or "\":" in t):
        return True
    return any(m in t for m in _JUNK_MARKERS)


def visible_texts(node):
    """Текстовые узлы без <script>/<style> и без JSON/JS-мусора (на Яндекс Маркете он лежит прямо в карточках)."""
    return [t for t in node.xpath(_TEXT_XPATH).getall() if not _is_junk_text(t)]


# Выполняется внутри страницы: innerText содержит только реально отображаемый текст
# (без скрытого JSON, скриптов и служебных блоков).
YANDEX_JS = r"""() => {
  const SKIP = ['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'TEXTAREA'];
  const key = (h) => h.split('?')[0];
  const linkSel = "a[href*='/card/'], a[href*='/product--']";
  // Только реально видимые текстовые узлы: по одному на строку, без скрытого JSON.
  const lines = (root) => {
    const out = [], w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = w.nextNode())) {
      const el = n.parentElement;
      if (!el || SKIP.includes(el.tagName)) continue;
      if (el.checkVisibility && !el.checkVisibility({checkVisibilityCSS: true, visibilityProperty: true})) continue;
      const t = n.nodeValue.replace(/\s+/g, ' ').trim();
      if (t) out.push(t);
    }
    return out;
  };
  const seen = new Set(), out = [];
  for (const a of document.querySelectorAll(linkSel)) {
    const url = key(a.href);
    if (seen.has(url)) continue;
    seen.add(url);
    let card = a;
    for (let i = 0; i < 8; i++) {
      const p = card.parentElement;
      if (!p) break;
      const others = new Set(Array.from(p.querySelectorAll(linkSel)).map(x => key(x.href)));
      if (others.size > 1) break;  // родитель уже содержит другие товары
      card = p;
      if (lines(card).join(' ').length > 80) break;
    }
    const img = card.querySelector('img');
    out.push({
      url,
      lines: lines(card),
      title: a.getAttribute('title') || a.getAttribute('aria-label') || '',
      alt: img ? (img.getAttribute('alt') || '') : '',
    });
  }
  return out;
}"""


READY_SELECTOR = {
    "wildberries": "article.product-card, [data-nm-id]",
    "ozon": "a[href*='/product/']",
    "yandexmarket": "a[href*='/product--'], a[href*='/card/'], [data-auto='snippet-title']",
}

# Прогрев сессии: сначала главная страница, потом каталог.
WARMUP_URL = {
    "ozon": "https://www.ozon.ru/",
    "yandexmarket": "https://market.yandex.ru/",
}

STORE_ALIASES = {"yandex": "yandexmarket", "market": "yandexmarket", "yandex_market": "yandexmarket"}


class EcommerceSpider(scrapy.Spider):
    name = "ecommerce"
    allowed_domains = [
        "ozon.ru", "www.ozon.ru", "wildberries.ru", "www.wildberries.ru",
        "market.yandex.ru", "yandex.ru",
    ]

    custom_settings = {
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "CONCURRENT_REQUESTS": 3,
        "DOWNLOAD_DELAY": 5.0,
        "DOWNLOAD_DELAY_JITTER": 0.5,
        "ROBOTSTXT_OBEY": False,
        "HTTPERROR_ALLOW_ALL": True,
        # Только https через Playwright: иначе стартуют два обработчика
        # и оба пытаются открыть один и тот же профиль браузера.
        "DOWNLOAD_HANDLERS": {
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        # Не подменять заголовки/User-Agent браузера заголовками Scrapy.
        "PLAYWRIGHT_PROCESS_REQUEST_HEADERS": None,
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 60_000,
        "PLAYWRIGHT_LAUNCH_OPTIONS": dict(BROWSER_OPTIONS),
        "PLAYWRIGHT_CONTEXTS": {
            "default": {
                **BROWSER_OPTIONS,
                "user_data_dir": PROFILE_DIR,  # cookies сохраняются между запусками
                "locale": "ru-RU",
                "no_viewport": True,
            }
        },
    }

    def __init__(self, category="ноутбук", stores="wildberries,ozon", max_items=110, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = (category or "ноутбук").strip()
        self.stores = [x.strip().lower() for x in stores.split(",") if x.strip()]
        self.stores = [STORE_ALIASES.get(x, x) for x in self.stores]
        self.stores = [x for x in dict.fromkeys(self.stores) if x in READY_SELECTOR]
        if not self.stores:
            raise ValueError("stores должен содержать wildberries, ozon и/или yandexmarket")
        self.max_items = max(1, int(max_items))
        self.collected = {s: 0 for s in self.stores}
        base, rem = divmod(self.max_items, len(self.stores))
        self.targets = {s: base + (1 if i < rem else 0) for i, s in enumerate(self.stores)}
        self.seen_urls = set()
        self.failures = {s: 0 for s in self.stores}  # подряд идущие ошибки загрузки по магазину

    async def start(self):
        for store in self.stores:
            if self.targets[store] > 0:
                yield self.make_request(store, 1, self.build_url(store, self.category, 1))

    def build_url(self, store, category, page):
        if store == "yandexmarket":
            params = {"text": category}
            if page > 1:
                params["page"] = page
            return "https://market.yandex.ru/search?" + urlencode(params)

        if store == "wildberries":
            params = {"search": category}
            if page > 1:
                params["page"] = page
            return "https://www.wildberries.ru/catalog/0/search.aspx?" + urlencode(params)

        if category.lower() in {"ноутбук", "ноутбуки"}:
            base = "https://www.ozon.ru/category/noutbuki-planshety-i-elektronnye-knigi-8730/"
            params = {
                "category_was_predicted": "true",
                "deny_category_prediction": "true",
                "from_global": "true",
                "text": category,
            }
            if page > 1:
                params["page"] = page
            return base + "?" + urlencode(params)

        params = {"text": category}
        if page > 1:
            params["page"] = page
        return "https://www.ozon.ru/search/?" + urlencode(params)

    def make_request(self, store, page, url):
        request_url = url
        methods = []
        if store in WARMUP_URL and page == 1:
            # Прогрев: сначала главная страница, потом каталог (как делает живой пользователь).
            request_url = WARMUP_URL[store]
            methods += [
                PageMethod("wait_for_timeout", 4000),
                PageMethod("evaluate", "window.scrollBy(0, 400)"),
                PageMethod("wait_for_timeout", 2000),
                PageMethod("goto", url, wait_until="domcontentloaded"),
            ]
        methods += [
            PageMethod("wait_for_load_state", "domcontentloaded"),
            # Ждём карточки до 60 сек. Если появилась проверка, пройдите её вручную в окне.
            PageMethod("wait_for_selector", READY_SELECTOR[store], timeout=60_000),
        ]
        # Несколько прокруток, чтобы подгрузились все карточки (ленивая загрузка).
        # У Яндекс Маркета карточки "оживают" только после прокрутки, поэтому там их больше и пауза дольше.
        scrolls, pause = (12, 1500) if store == "yandexmarket" else (5, 1200)
        for _ in range(scrolls):
            methods += [
                PageMethod("evaluate", "window.scrollBy(0, Math.max(window.innerHeight * 0.9, 700))"),
                PageMethod("wait_for_timeout", pause),
            ]
        if store == "yandexmarket":
            methods += [PageMethod("wait_for_timeout", 2000), PageMethod("evaluate", YANDEX_JS)]
        return scrapy.Request(
            request_url,
            callback=self.parse,
            errback=self.errback_marketplace,
            dont_filter=True,
            meta={
                "playwright": True,
                "playwright_page_methods": methods,
                "store": store,
                "page_number": page,
            },
            # Заголовки не задаём: браузер использует собственные.
        )

    def parse(self, response):
        store = response.meta["store"]
        page = response.meta["page_number"]
        self.logger.info("[%s] page=%d status=%d url=%s", store.upper(), page, response.status, response.url)

        parsers = {
            "wildberries": self.parse_wildberries,
            "ozon": self.parse_ozon,
            "yandexmarket": self.parse_yandexmarket,
        }
        items = list(parsers[store](response))
        for it in items:
            it["rating"] = self.norm_rating(it.get("rating"))

        if not items:
            blocked = response.status in {403, 429, 498, 503} or self.is_block_page(response, store)
            self.logger.warning(
                "[%s] товаров нет (status=%d, блокировка=%s). HTML сохранён в debug_%s_%d.html",
                store.upper(), response.status, blocked, store, page,
            )
            Path(f"debug_{store}_{page}.html").write_text(response.text, encoding="utf-8")
            return

        self.failures[store] = 0
        new_count = 0
        for item in items:
            if self.collected[store] >= self.targets[store]:
                break
            url = item.get("url")
            if not item.get("title") or not url or url in self.seen_urls:
                continue
            self.seen_urls.add(url)
            self.collected[store] += 1
            new_count += 1
            yield ProductItem(**item)

        self.logger.info(
            "[%s] page=%d parsed=%d new=%d total=%d/%d",
            store.upper(), page, len(items), new_count, self.collected[store], self.targets[store],
        )

        if self.collected[store] < self.targets[store] and page < 30:
            yield self.make_request(store, page + 1, self.build_url(store, self.category, page + 1))

    def is_block_page(self, response, store):
        text = " ".join(visible_texts(response)).lower()
        title = (response.css("title::text").get() or "").lower()
        markers = {
            "ozon": ["доступ ограничен", "похоже, нет соединения", "выключите vpn",
                     "обновить страницу", "antibot", "captcha"],
            "wildberries": ["проверяем браузер", "подозрительная активность",
                            "попробуйте обновить страницу", "captcha"],
            "yandexmarket": ["smartcaptcha", "вы не робот", "подтвердите, что запросы",
                             "showcaptcha", "captcha"],
        }
        return any(m in text or m in title for m in markers[store])

    def parse_wildberries(self, response):
        cards = response.css("article.product-card, .product-card, [data-nm-id], [class*='product-card']")
        self.logger.info("[WB] candidate cards=%d", len(cards))
        for card in cards:
            title = self.first_text(card, [
                ".product-card__name::text", "[class*='product-card__name']::text",
                "[class*='name']::text", "a[aria-label]::attr(aria-label)", "a::text"
            ])
            href = card.css("a[href*='/catalog/']::attr(href)").get() or card.css("a[href]::attr(href)").get()
            price = (
                self.first_text(card, [".price__lower-price::text", ".price::text", "[class*='price']::text"])
                or self.price_from_text(card)
            )
            rating = self.first_text(card, ["[class*='rating']::text", "[class*='product-card__rating']::text"])
            description = self.first_text(card, [".product-card__brand::text", "[class*='brand']::text"])
            if title and href:
                yield {
                    "title": title, "price": price, "rating": rating,
                    "description": description or "", "url": response.urljoin(href), "source": "Wildberries"
                }

    def parse_ozon(self, response):
        links = response.css("a[href*='/product/']")
        self.logger.info("[OZON] product links=%d", len(links))
        local_urls = set()
        for link in links:
            href = link.css("::attr(href)").get()
            if not href:
                continue
            url = response.urljoin(href).split("?")[0]  # без трекинговых параметров
            if url in local_urls:
                continue
            local_urls.add(url)
            card = link
            for _ in range(6):
                parent = card.xpath("..")
                if not parent:
                    break
                card = parent
                if len(" ".join(visible_texts(card)).strip()) > 80:
                    break
            title = self.card_title(card, link)
            price = (
                self.first_text(card, ["[class*='price']::text", "[data-widget*='price']::text"])
                or self.price_from_text(card)
            )
            rating = (
                self.first_text(card, ["[class*='rating']::text", "[class*='stars']::text"])
                or self.rating_from_text(card)
            )
            description = " ".join(visible_texts(card)).strip()[:500]
            if title and len(title) >= 5:
                yield {
                    "title": title, "price": price, "rating": rating,
                    "description": description, "url": url, "source": "Ozon"
                }

    BADGE_WORDS = (
        "распродажа", "вау-цены", "скидки недели", "морковок", "цена что надо",
        "осталось", "бренд проверен", "реклама", "отзыв", "оценк", "доставка", "в корзину",
        "купил", "сравнить", "кешбэк", "баллов", "рассрочка",
    )

    @classmethod
    def is_title_like(cls, text):
        """Похоже ли на название товара, а не на бейдж ('Распродажа', 'Вау-цены'...) или цену."""
        if len(text) < 15 or "₽" in text or "%" in text:
            return False
        low = text.lower()
        return not any(w in low for w in cls.BADGE_WORDS)

    @classmethod
    def card_title(cls, card, link):
        # 1) текст самой ссылки; 2) другие ссылки на товар в карточке; 3) любые текстовые узлы
        candidates = [" ".join(visible_texts(link))]
        candidates += [" ".join(visible_texts(a)) for a in card.css("a[href*='/product'], a[href*='/card/']")]
        candidates += link.xpath("@title | @aria-label").getall()
        candidates += card.xpath(".//img/@alt").getall()
        candidates += visible_texts(card)
        for raw in candidates:
            text = " ".join(raw.split())
            if cls.is_title_like(text):
                return text
        return None

    @staticmethod
    def rating_from_text(node):
        """Запасной вариант: отдельный текстовый узел вида '4.8' или '4,8'."""
        for raw in visible_texts(node):
            text = raw.strip()
            if re.fullmatch(r"[1-5][.,]\d", text):
                return text
        return None

    @staticmethod
    def find_card(link):
        """Переход от ссылки к контейнеру карточки (где уже достаточно текста)."""
        card = link
        for _ in range(6):
            parent = card.xpath("..")
            if not parent:
                break
            card = parent
            if len(" ".join(visible_texts(card)).strip()) > 80:
                break
        return card

    @staticmethod
    def norm_rating(value):
        """Единый формат рейтинга - десятичная запятая"""
        return value.replace(".", ",") if value else value

    def parse_yandexmarket(self, response):
        data = None
        for pm in response.meta.get("playwright_page_methods", []):
            if getattr(pm, "method", None) == "evaluate" and pm.args and pm.args[0] == YANDEX_JS:
                data = getattr(pm, "result", None)
        if not data:
            self.logger.warning("[YANDEX] нет результата JS-извлечения, разбираю HTML")
            yield from self.parse_yandexmarket_html(response)
            return

        self.logger.info("[YANDEX] cards from page=%d", len(data))
        for row in data:
            lines = [" ".join(x.split()) for x in (row.get("lines") or []) if x.strip()]
            if "Реклама" in lines:
                continue  # рекламные вставки часто не относятся к запросу
            title = next((x for x in lines if self.is_title_like(x)), None)
            if not title:
                for extra in (row.get("title"), row.get("alt")):
                    extra = " ".join((extra or "").split())
                    if self.is_title_like(extra):
                        title = extra
                        break
            rating = next((x for x in lines if re.fullmatch(r"[1-5][.,]\d", x)), None)
            if title and len(title) >= 5:
                yield {
                    "title": title,
                    "price": next((self.price_from_string(x) for x in lines if "₽" in x), None),
                    "rating": rating,
                    "description": " ".join(lines)[:500],
                    "url": response.urljoin(row["url"]),
                    "source": "Yandex Market",
                }

    def parse_yandexmarket_html(self, response):
        links = response.css("a[href*='/product--'], a[href*='/card/']")
        self.logger.info("[YANDEX] product links=%d", len(links))
        local_urls = set()
        for link in links:
            href = link.css("::attr(href)").get()
            if not href:
                continue
            url = response.urljoin(href).split("?")[0]
            if url in local_urls:
                continue
            local_urls.add(url)
            card = self.find_card(link)
            if any(t.strip() == "Реклама" for t in visible_texts(card)):
                continue  # рекламные вставки часто не относятся к запросу
            title = (
                self.first_text(card, ["[data-auto='snippet-title']::text", "[data-auto='snippet-title-header']::text",
                                       "h3::text"])
                or self.card_title(card, link)
            )
            price = (
                self.first_text(card, ["[data-auto='snippet-price-current']::text", "[data-auto='price-value']::text"])
                or self.price_from_text(card)
            )
            rating = (
                self.first_text(card, ["[data-auto='rating-badge'] ::text", "[data-auto='reviews'] ::text"])
                or self.rating_from_text(card)
            )
            description = " ".join(visible_texts(card)).strip()[:500]
            if title and len(title) >= 5:
                yield {
                    "title": title, "price": price, "rating": rating,
                    "description": description, "url": url, "source": "Yandex Market"
                }

    @staticmethod
    def price_from_string(text):
        m = re.search(r"(\d[\d\s\u00a0\u2009\u202f]*)\s*₽", text)
        return (" ".join(m.group(1).split()) + " ₽") if m else None

    @classmethod
    def price_from_text(cls, node):
        """Запасной вариант: первая сумма вида '12 345 ₽' в тексте карточки."""
        return cls.price_from_string(" ".join(visible_texts(node)))

    @staticmethod
    def first_text(node, selectors):
        for selector in selectors:
            value = node.css(selector).get()
            if value:
                value = " ".join(value.split())
                if value:
                    return value
        return None

    def errback_marketplace(self, failure):
        request = failure.request
        store = request.meta.get("store", "?")
        page = request.meta.get("page_number", 0)
        self.logger.error("[%s] page=%s request failed: %s", store.upper(), page, failure.value)

        # Одна упавшая страница (таймаут, капча) не должна обрывать весь магазин:
        # пробуем следующую страницу, но не больше 2 неудач подряд.
        if store not in self.collected:
            return
        self.failures[store] += 1
        if self.failures[store] <= 2 and self.collected[store] < self.targets[store] and page < 30:
            yield self.make_request(store, page + 1, self.build_url(store, self.category, page + 1))