from typing import List
from langchain_core.tools import StructuredTool

from ..context import CovalentContext
from ..types import CovalentConfig
from .api_info_tool import create_covalent_api_info_tool
from .query_tool import (
    create_covalent_query_tool,
    create_covalent_result_head_tool,
    create_covalent_result_jq_tool
)


def create_covalent_tools(
    config: CovalentConfig,
    context: CovalentContext,
) -> List[StructuredTool]:
    return [
        create_covalent_api_info_tool(),
        create_covalent_query_tool(config, context),
        create_covalent_result_head_tool(context),
        create_covalent_result_jq_tool(context),
    ]
