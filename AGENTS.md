# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in this LangChain API repository.

## Project Overview

This is a FastAPI-based LangChain microservice providing RAG (Retrieval-Augmented Generation) and Agent capabilities with OpenAI-compatible API endpoints. The project uses modern Python patterns with strong typing and modular architecture.

## Development Commands

### Package Management
```bash
# Install dependencies
uv sync

# Run main application
uv run python main.py

# Run agent demo
uv run python langchain_api/agent/main.py

# Run RAG demo
uv run python langchain_api/rag/main.py
```

### Testing
**Note**: No test framework currently configured. Recommended to add pytest:
```bash
# Install pytest (if adding tests)
uv add --dev pytest pytest-asyncio

# Run tests (once configured)
uv run pytest
```

### Code Quality
**Note**: No linting configured. Recommended to add ruff:
```bash
# Install ruff (if adding linting)
uv add --dev ruff black

# Run linting and formatting
uv run ruff check .
uv run ruff format .
```

## Code Style Guidelines

### Import Order
```python
# 1. Standard library imports
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, NotRequired, TypedDict, Literal

# 2. Third-party imports
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_openai import ChatOpenAI
from loguru import logger

# 3. Local imports
from langchain_api.middleware import PlanningMiddleware
from langchain_api.settting import settings
```

### Type System
- Use extensive type annotations with `typing` module
- Prefer `TypedDict`, `NotRequired`, `Literal` for complex types
- All function signatures must have type hints
- Use Pydantic models for configuration and data validation

### Naming Conventions
- **Classes**: PascalCase (`Agent`, `Settings`, `CustomState`)
- **Functions/Variables**: snake_case (`add`, `multiply`, `get_settings`)
- **Constants**: UPPER_SNAKE_CASE (`RAG_SYSTEM_PROMPT`, `REWRITE_QUREY_PROMPT`)
- **Private members**: Prefix with underscore (`_internal_method`)

### Code Organization
- **Modular design**: Separate modules for different functionalities
- **Middleware pattern**: Implement custom middleware for cross-cutting concerns
- **Factory pattern**: Use factory functions for configuration (`get_settings()`)
- **Composition over inheritance**: Prefer composition for middleware and agents

### Documentation Style
- **Docstrings**: Use Chinese documentation with comprehensive parameter descriptions
- **Comments**: Mix of Chinese and English is acceptable
- **Prompt templates**: Use multi-line strings for AI prompts with clear variable placeholders

### Error Handling
- Use `assert` statements for configuration validation during development
- Raise descriptive exceptions for invalid configurations
- Prefer built-in exceptions over custom exception classes (unless complex error handling needed)
- Always include error messages in Chinese for user-facing errors

## Architecture Patterns

### Middleware Architecture
- Implement middleware classes inheriting from LangChain middleware base classes
- Middleware should be composable and chainable
- Use middleware for cross-cutting concerns like RAG, planning, logging

### Agent Design
- Use tool-based agents with Python functions as tools
- Implement memory using LangGraph checkpointer for conversation persistence
- Support streaming responses for real-time interaction

### RAG System
- Implement multiple retrieval strategies (BM25, DenseVector)
- Include query rewriting and enhancement logic
- Add intelligent routing between RAG and direct LLM responses

### Configuration Management
- Use Pydantic BaseSettings for type-safe configuration
- Support environment-based configuration with `.env` files
- Centralize all settings in `langchain_api/settting.py` (note: typo exists)
- Validate configuration at startup

## File Structure Conventions

```
langchain_api/
├── agent/              # Agent-related functionality
│   ├── agent.py        # Agent implementation
│   └── main.py         # Agent demo/entry point
├── rag/                # RAG-related functionality
│   └── main.py         # RAG demo/entry point
├── middleware.py       # Custom middleware implementations
├── retriever.py        # Vector store and retrieval logic
└── settting.py         # Configuration management (note: typo)
```

## Development Guidelines

### When Adding New Features
1. Follow the existing modular structure
2. Add proper type annotations for all new code
3. Include comprehensive docstrings in Chinese
4. Implement appropriate error handling
5. Add configuration options to settings if needed

### When Modifying Existing Code
1. Maintain existing code style and patterns
2. Preserve Chinese documentation and comments
3. Update type annotations if changing function signatures
4. Test with the demo scripts in agent/ and rag/ directories

### Configuration Management
- All new configuration should go in `langchain_api/settting.py`
- Use environment variables for sensitive data (API keys, etc.)
- Validate configuration at application startup
- Provide sensible defaults where possible

## Dependencies

Core dependencies are managed via `uv` and defined in `pyproject.toml`:
- FastAPI for web framework
- LangChain for AI/ML functionality
- Elasticsearch for vector storage
- Loguru for logging
- Pydantic for data validation

## Known Issues

1. **Typo**: `settting.py` should be `settings.py` - maintain compatibility when fixing
2. **No testing framework**: Consider adding pytest for test coverage
3. **No linting configuration**: Consider adding ruff for code quality
4. **Empty README**: Documentation needs completion

## Testing Strategy

When adding tests (recommended):
- Use pytest with async support for FastAPI endpoints
- Test middleware components independently
- Include integration tests for agent and RAG functionality
- Mock external dependencies (OpenAI API, Elasticsearch)