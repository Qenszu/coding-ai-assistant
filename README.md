# Coding AI Assistant

A collection of Python AI agents for researching developer tools and scraping the web. Both agents use **Google Gemini** for reasoning and **Firecrawl** for search and page extraction.

## Projects

### `simple_agent` — Interactive Firecrawl MCP Agent

A conversational agent that connects to the [Firecrawl MCP server](https://github.com/mendableai/firecrawl-mcp) via LangChain. Ask it to scrape, crawl, or extract data from websites in a chat loop.

**Stack:** LangChain, LangGraph, MCP, Gemini

### `advanced_agent` — Developer Tools Research Agent

A multi-step [LangGraph](https://langchain-ai.github.io/langgraph/) workflow that researches and compares developer tools for a given query. It:

1. Searches for comparison articles and extracts tool names
2. Researches each tool (pricing, tech stack, API, integrations)
3. Generates a concise recommendation

**Stack:** LangGraph, Firecrawl Python SDK, Gemini, Pydantic

## Quick Start

Create a `.env` file in the agent directory you want to run:

```env
GEMINI_API_KEY=your_gemini_key
FIRECRAWL_API_KEY=your_firecrawl_key
```

### Simple Agent

```bash
cd simple_agent
uv sync
uv run main.py
```

Type your questions at the prompt. Enter `q` to quit.

### Advanced Agent

```bash
cd advanced_agent
uv sync
uv run main.py
```

Enter a developer-tools query (e.g. *"best backend-as-a-service for Python"*). Type `quit` or `exit` to stop.

## Project Structure

```
coding-ai-assistant/
├── simple_agent/          # MCP-based chat agent
│   └── main.py
├── advanced_agent/        # LangGraph research workflow
│   ├── main.py
│   └── src/
│       ├── workflow.py    # Graph: extract → research → analyze
│       ├── firecrawl.py   # Firecrawl search & scrape wrapper
│       ├── models.py      # Pydantic state & company models
│       └── prompts.py     # LLM prompt templates
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
