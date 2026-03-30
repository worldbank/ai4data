"""ai4data.metadata.reviewer — async AI-powered metadata reviewer.

Install with one of:
    uv pip install ai4data[metadata_reviewer,openai]
    uv pip install ai4data[metadata_reviewer,azure]
    uv pip install ai4data[metadata_reviewer,ollama]
    uv pip install ai4data[metadata_reviewer,anthropic]

Quickstart (OpenAI)
-------------------
    from ai4data.metadata.reviewer import MetadataReviewerClient

    client = MetadataReviewerClient.from_openai(model="gpt-5", api_key="sk-...")
    job = client.submit(metadata_dict)
    result = job.wait_sync(timeout=300)

Quickstart (Azure OpenAI)
-------------------------
    from azure.identity import get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        ...
    )
    client = MetadataReviewerClient.from_azure(
        model="gpt-5",
        azure_endpoint="https://<resource>.openai.azure.com/",
        azure_deployment="<deployment>",
        api_version="2024-02-01",
        azure_ad_token_provider=token_provider,
    )

Quickstart (Ollama)
-------------------
    client = MetadataReviewerClient.from_ollama(model="llama3.2", port=11434)

Quickstart (Anthropic)
----------------------
    client = MetadataReviewerClient.from_anthropic(model="claude-sonnet-4-6", api_key="sk-ant-...")
"""

from .client import MetadataReviewerClient
from .jobs import Job, JobStatus

__all__ = ["MetadataReviewerClient", "Job", "JobStatus"]
