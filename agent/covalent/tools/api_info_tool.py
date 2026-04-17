from typing import Optional
from pydantic import BaseModel, Field
from loguru import logger
from langchain_core.tools import StructuredTool

from ..api_spec import get_initial_api_info, get_category_doc


class ApiInfoInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description='Optional category to get detailed docs. Options: workflows, balances, transactions, nft-security-crosschain, utility. Leave empty for overview.'
    )


def create_covalent_api_info_tool() -> StructuredTool:
    def func(category: Optional[str] = None) -> str:
        try:
            logger.info(f'Executing Covalent API info tool with category={category}')

            if category:
                doc = get_category_doc(category)
                logger.info(f'Returned category documentation for {category}, length={len(doc)}')
                return f"""📖 {category.upper()} DOCUMENTATION:

{doc}

💡 This includes the shared workflows reference plus the detailed documentation for "{category}".
   Call with another category if needed, or use covalent_query to make requests."""

            initial_info = get_initial_api_info()
            logger.info(f'Returned initial API info, length={len(initial_info)}')

            return f"""📖 COVALENT API OVERVIEW:

{initial_info}

⚠️ CRITICAL REMINDERS:
- Chain names are CASE-SENSITIVE: "eth-mainnet" not "ethereum"
- ENS names are supported for eth-mainnet (e.g., vitalik.eth)
- Balance values need division by 10^contract_decimals for display
- Page numbers are 0-indexed (first page is 0)

🚀 NEXT STEPS:
1. If you need more endpoint details, call with category (e.g., "balances", "transactions")
2. When ready, use covalent_query to execute requests"""
        except Exception as error:
            logger.error(f'Error executing API info tool: {error}')
            return f'Error reading API spec: {str(error)}'

    return StructuredTool(
        name='covalent_api_info',
        description="""Get the Covalent REST API documentation.

FIRST CALL (no category): Returns overview, chain names, common workflows, and available categories.
SUBSEQUENT CALLS (with category): Returns detailed endpoint docs for that category while keeping the shared workflows reference in context.

Categories:
- "workflows" - Common usage patterns (DEFAULT if no category)
- "balances" - Token balances, transfers, holders, portfolio
- "transactions" - Transaction history, blocks, summaries
- "nft-security-crosschain" - NFTs, approvals, multi-chain activity
- "utility" - Pricing, gas, events, chains status

Usage:
1. Call without category to understand available endpoints and workflows
2. Call with specific category if you need detailed endpoint parameters""",
        func=func,
        args_schema=ApiInfoInput
    )
