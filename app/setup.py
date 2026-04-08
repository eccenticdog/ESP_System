from setuptools import setup, find_packages

setup(
    name="its_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "pydantic",
        "pydantic-settings",
        "openai",
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "langgraph-checkpoint-postgres",
        "langgraph-supervisor",
        "langmem",
        "mcp",
        "httpx-sse",
        "psycopg[binary]",
        "pymysql",
        "dbutils",
        "pystun3"
    ],
)
