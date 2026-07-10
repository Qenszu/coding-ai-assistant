import os
from dotenv import load_dotenv
from firecrawl import Firecrawl
from firecrawl.v2.types import ScrapeOptions

load_dotenv()

class FirecrawlService:
    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("Missing firecrawl api key")

        self.app = Firecrawl(api_key=api_key)

    def search_companies(self, query: str, num_results: int = 5):
        try:
            result = self.app.search(
                query=f"{query} company pricing",
                limit=num_results,
                scrape_options=ScrapeOptions(
                    formats=["markdown"],
                )
            )
            return result
        except Exception as e:
            print("Search company error: ", e)
            return []

    def scrape_company_pages(self, url: str):
        try:
            result = self.app.scrape(
                url=url, 
                formats=["markdown"]
            )
            return result
        except Exception as e:
            print("Company pages error: ", e)
            return None