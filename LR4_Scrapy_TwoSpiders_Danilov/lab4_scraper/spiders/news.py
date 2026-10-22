import scrapy

from lab4_scraper.items import NewsItem


class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["habr.com"]
    start_urls = ["https://habr.com/ru/articles/"]

    def __init__(self, keywords=None, days=7, max_items=120, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keywords = [x.strip().lower() for x in (keywords or "").split(",") if x.strip()]
        self.days = int(days)
        self.max_items = int(max_items)
        self.collected = 0

    def parse(self, response):
        articles = response.css("article.tm-articles-list__item")
        if not articles:
            # Запасной селектор на случай небольших изменений вёрстки.
            articles = response.css("article")

        for article in articles:
            # Заголовок находится во вложенных элементах,
            # поэтому XPath здесь надёжнее обычного ::text.
            title = article.xpath("string(.//a[contains(@class, 'tm-title__link')])").get()
            if not title:
                title = article.xpath("string(.//h2)").get()

            link = article.css("a.tm-title__link::attr(href)").get()
            if not link:
                link = article.xpath(".//h2//a/@href").get()

            summary = " ".join(article.css(".tm-article-snippet__lead *::text").getall())
            if not summary:
                summary = " ".join(article.css("p.tm-article-snippet__lead-paragraph::text").getall())

            date = article.css("time::attr(datetime)").get()
            if not date:
                date = article.css("time::text").get()

            if not title or not link:
                self.logger.warning("Could not parse article on %s", response.url)
                continue

            title_clean = title.strip()
            if self.keywords:
                haystack = f"{title_clean} {summary}".lower()
                if not any(keyword in haystack for keyword in self.keywords):
                    continue

            self.collected += 1
            yield NewsItem(
                title=title_clean,
                summary=summary,
                date=date,
                url=response.urljoin(link),
                source="Habr",
            )

            if self.collected >= self.max_items:
                return

        if self.collected >= self.max_items:
            return

        # Формируем адрес следующей страницы напрямую,
        current_page = 1
        marker = "/page"
        if marker in response.url:
            try:
                current_page = int(response.url.split("/page")[1].split("/")[0])
            except (ValueError, IndexError):
                current_page = 1

        next_page = current_page + 1
        next_url = f"https://habr.com/ru/articles/page{next_page}/"

        if self.collected < self.max_items:
            yield scrapy.Request(
                next_url,
                callback=self.parse,
                dont_filter=True
            )