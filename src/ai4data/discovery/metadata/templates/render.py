import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ...config import EMBEDDING_CONTENT_TEMPLATES_PATH


def get_searchpath(metadata_type: str, searchpath: str = None):
    """
    Get the searchpath for a metadata type.

    Args:
        metadata_type (str): The type of metadata.
        searchpath (str): The path to search for the Jinja2 template.

    Returns:
        str: The searchpath.
    """
    searchpath = searchpath or EMBEDDING_CONTENT_TEMPLATES_PATH
    searchpath = os.path.join(searchpath, metadata_type)

    return searchpath


def render_embedding_content(metadata: dict, metadata_type: str, field: str, searchpath: str = None):
    """
    Render the content for a metadata type using a Jinja2 template.
    The content is used to generate vector embeddings for the metadata.
    The template is loaded from the ``searchpath``, which defaults to
    ``EMBEDDING_CONTENT_TEMPLATES_PATH`` from ``ai4data.discovery.config``
    (override with env ``AI4DATA_EMBEDDING_CONTENT_TEMPLATES_PATH``) if not provided explicitly.

    Args:
        metadata (dict): The metadata to render.
        metadata_type (str): The type of metadata.
        field (str): The field to render.
        searchpath (str): The path to search for the Jinja2 template.

    Returns:
        str: The rendered content.
    """

    # Load the Jinja2 environment
    searchpath = get_searchpath(metadata_type, searchpath)

    env = Environment(loader=FileSystemLoader(searchpath=searchpath), autoescape=select_autoescape())

    template_name = f"{metadata_type}__{field}__page_content.jinja2"

    # Load the template
    template = env.get_template(template_name)

    # Render the template with metadata
    page_content = template.render(metadata)

    return page_content
