from typing import Any, Dict, List, Optional, Union

from firecrawl.v2.types import Document, SearchData, SearchResultWeb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from .firecrawl import FirecrawlService
from .models import CompanyAnalysis, CompanyInfo, ResearchState
from .prompts import DeveloperToolsPrompts

SearchResult = Union[SearchResultWeb, Document]


class Workflow:
    def __init__(self):
        self.firecrawl = FirecrawlService()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
        )
        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(ResearchState)
        graph.add_node("extract_tools", self._extract_tools_step)
        graph.add_node("research", self._research_step)
        graph.add_node("analyze", self._analyze_step)

        graph.set_entry_point("extract_tools")
        graph.add_edge("extract_tools", "research")
        graph.add_edge("research", "analyze")
        graph.add_edge("analyze", END)

        return graph.compile()

    @staticmethod
    def _iter_web_results(search_results: Union[SearchData, List[Any]]) -> List[SearchResult]:
        if isinstance(search_results, SearchData):
            return search_results.web or []
        return []

    @staticmethod
    def _get_url(result: SearchResult) -> str:
        if isinstance(result, Document):
            if result.metadata and result.metadata.url:
                return result.metadata.url
            return ""
        return result.url or ""

    @staticmethod
    def _get_title(result: SearchResult) -> str:
        if isinstance(result, Document):
            if result.metadata and result.metadata.title:
                return result.metadata.title
            return "Unknown"
        return result.title or "Unknown"

    @staticmethod
    def _get_markdown(result: SearchResult) -> str:
        if isinstance(result, Document) and result.markdown:
            return result.markdown
        return ""

    def _collect_content(self, search_results: Union[SearchData, List[Any]], max_chars: int = 1500) -> str:
        all_content = ""

        for result in self._iter_web_results(search_results):
            markdown = self._get_markdown(result)
            if not markdown:
                url = self._get_url(result)
                if not url:
                    continue
                scraped = self.firecrawl.scrape_company_pages(url)
                if scraped and scraped.markdown:
                    markdown = scraped.markdown

            if markdown:
                all_content += markdown[:max_chars] + "\n\n"

        return all_content

    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        print(f"🔍 Finding articles about: {state.query}")

        article_query = f"{state.query} tools comparison beat alternatives"
        search_results = self.firecrawl.search_companies(article_query, num_results=3)
        all_content = self._collect_content(search_results)

        messages = [
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(state.query, all_content)),
        ]

        try:
            response = self.llm.invoke(messages)
            tool_names = [
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip()
            ]
            print(f"Extracted tools: {', '.join(tool_names[:5])}")
            return {"extracted_tools": tool_names}
        except Exception as e:
            print("Extracting tools error: ", e)
            return {"extracted_tools": []}

    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis:
        structured_llm = self.llm.with_structured_output(CompanyAnalysis)

        messages = [
            SystemMessage(content=self.prompts.TOOL_ANALYSIS_SYSTEM),
            HumanMessage(content=self.prompts.tool_analysis_user(company_name, content)),
        ]

        try:
            return structured_llm.invoke(messages)
        except Exception as e:
            print("Analyze company error: ", e)
            return CompanyAnalysis(
                pricing_model="Unknown",
                is_open_source=None,
                tech_stack=[],
                description="Failed",
                language_support=[],
                integration_capabilities=[],
            )

    def _build_company_from_result(
        self,
        tool_name: str,
        result: SearchResult,
    ) -> Optional[CompanyInfo]:
        url = self._get_url(result)
        if not url:
            return None

        company = CompanyInfo(
            name=tool_name,
            description=self._get_markdown(result) or self._get_title(result),
            website=url,
            tech_stack=[],
            competitors=[],
        )

        content = self._get_markdown(result)
        if not content:
            scraped = self.firecrawl.scrape_company_pages(url)
            if scraped and scraped.markdown:
                content = scraped.markdown

        if content:
            analysis = self._analyze_company_content(company.name, content)
            company.pricing_model = analysis.pricing_model
            company.is_open_source = analysis.is_open_source
            company.tech_stack = analysis.tech_stack
            company.description = analysis.description
            company.api_available = analysis.api_available
            company.language_support = analysis.language_support
            company.integration_capabilities = analysis.integration_capabilities

        return company

    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        extracted_tools = state.extracted_tools or []

        if not extracted_tools:
            print("⚠️ No extracted tools found, falling back to direct search")
            search_results = self.firecrawl.search_companies(state.query, num_results=4)
            tool_names = [
                self._get_title(result)
                for result in self._iter_web_results(search_results)
            ]
        else:
            tool_names = extracted_tools[:4]

        print(f"🔬 Researching specific tools: {', '.join(tool_names)}")

        companies: List[CompanyInfo] = []
        for tool_name in tool_names:
            if not tool_name or tool_name == "Unknown":
                continue

            tool_search_results = self.firecrawl.search_companies(
                f"{tool_name} official site",
                num_results=1,
            )
            web_results = self._iter_web_results(tool_search_results)
            if not web_results:
                continue

            company = self._build_company_from_result(tool_name, web_results[0])
            if company:
                companies.append(company)

        return {"companies": companies}

    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        print("Generating recommendations")

        if not state.companies:
            return {"analysis": "No tools were found to compare. Try a more specific query."}

        company_data = ", ".join(
            company.model_dump_json() for company in state.companies
        )

        messages = [
            SystemMessage(content=self.prompts.RECOMMENDATIONS_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(state.query, company_data)),
        ]

        response = self.llm.invoke(messages)
        return {"analysis": response.content}

    def run(self, query: str) -> ResearchState:
        initial_state = ResearchState(query=query)
        final_state = self.workflow.invoke(initial_state)
        return ResearchState(**final_state)
