from setuptools import setup, find_packages

setup(
    name="rag-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "chromadb",
        "pyyaml",
        "beautifulsoup4",
        "langdetect",
        "pydantic>=2.0.0",
        "ollama",
        "numpy",
        "aiohttp",
        "typing-extensions",
        "async-timeout",
    ],
    python_requires=">=3.8",
)