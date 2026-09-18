import scrapy

from metals_scraper.items import BookItem


class BooksSpider(scrapy.Spider):
    name = "books"

    start_urls = [
        "https://books.toscrape.com/"
    ]

    def parse(self, response):
        for book in response.css("article.product_pod"):
            title = book.css("h3 a::attr(title)").get()
            price = book.css("p.price_color::text").get()
            availability = book.css("p.instock.availability::text").getall()
            rating = book.css("p.star-rating::attr(class)").get()

            relative_url = book.css("h3 a::attr(href)").get()
            url = response.urljoin(relative_url) if relative_url else None

            yield BookItem(
                title=title or "Не указано",
                price=price or "Не указано",
                availability=" ".join(x.strip() for x in availability if x.strip()),
                rating=rating or "Не указано",
                url=url or response.url,
            )

        next_page = response.css("li.next a::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
