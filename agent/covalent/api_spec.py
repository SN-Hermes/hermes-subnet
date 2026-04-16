from .references.workflows import workflows_reference
from .references.endpoints_balances import balances_reference
from .references.endpoints_transactions import transactions_reference
from .references.endpoints_nft_security_crosschain import nft_security_crosschain_reference
from .references.endpoints_utility import utility_reference


CATEGORY_MAP = {
    'workflows': {
        'content': workflows_reference,
        'title': 'Common Workflows'
    },
    'balances': {
        'content': balances_reference,
        'title': 'Balance Endpoints'
    },
    'transactions': {
        'content': transactions_reference,
        'title': 'Transaction Endpoints'
    },
    'nft-security-crosschain': {
        'content': nft_security_crosschain_reference,
        'title': 'NFT, Security & Cross-Chain Endpoints'
    },
    'utility': {
        'content': utility_reference,
        'title': 'Utility Endpoints'
    }
}


def get_workflows_doc() -> str:
    return workflows_reference


def get_endpoint_categories() -> str:
    return """## Available Endpoint Categories

Call this tool with a specific category to get detailed documentation:

| Category | Description |
|----------|-------------|
| `workflows` | Common usage patterns and best practices |
| `balances` | Token balances, transfers, holders, portfolio |
| `transactions` | Transaction history, blocks, summaries |
| `nft-security-crosschain` | NFTs, approvals, multi-chain activity |
| `utility` | Pricing, gas, events, chains status |

Example: Call with `category: "balances"` to get balance endpoint docs."""


def get_chain_names_ref() -> str:
    return """## Chain Names (CASE-SENSITIVE)

| Common Name | Chain Name | Chain ID |
|-------------|------------|----------|
| Ethereum | eth-mainnet | 1 |
| Polygon | matic-mainnet | 137 |
| Base | base-mainnet | 8453 |
| BSC | bsc-mainnet | 56 |
| Arbitrum | arbitrum-mainnet | 42161 |
| Optimism | optimism-mainnet | 10 |
| Avalanche | avalanche-mainnet | 43114 |
| Bitcoin | btc-mainnet | 20090103 |
| Solana | solana-mainnet | 1399811149 |
| Fantom | fantom-mainnet | 250 |
| zkSync Era | zksync-mainnet | 324 |
| Linea | linea-mainnet | 59144 |
| Scroll | scroll-mainnet | 534352 |
| Mantle | mantle-mainnet | 5000 |

## Common Query Parameters

| Parameter | Description |
|-----------|-------------|
| quote-currency | USD, CAD, EUR, SGD, INR, JPY, VND, CNY, KRW, RUB, TRY, NGN, ARS, AUD, CHF, GBP |
| page-size | Number of items (default 100) |
| page-number | 0-indexed page number |
| no-spam | true to filter spam tokens |

## Important Notes

- **Balance values**: Raw strings - divide by 10^contract_decimals for human-readable
- **ENS names**: Supported for eth-mainnet (e.g., vitalik.eth)
- **Page numbers**: 0-indexed (first page is 0)"""


def get_category_doc(category: str) -> str:
    normalized_category = category.strip().lower()
    mapping = CATEGORY_MAP.get(normalized_category)
    
    if not mapping:
        return f"Unknown category: {category}. Available: {', '.join(CATEGORY_MAP.keys())}"

    workflows = get_workflows_doc()
    content = mapping['content']

    if normalized_category == 'workflows':
        return f"# {mapping['title']}\n\n{content}"

    return f"""# Shared Workflow Rules

Keep this workflow reference in context while using the category-specific endpoint docs below.

{workflows}

---

# {mapping['title']}

{content}"""


def get_initial_api_info() -> str:
    workflows = get_workflows_doc()
    categories = get_endpoint_categories()
    chains = get_chain_names_ref()

    return f"""# Covalent (GoldRush) Blockchain Data API

{chains}

---

{categories}

---

# Common Workflows

{workflows}

---

💡 TIP: Call this tool again with a specific category (e.g., `category: "balances"`) to get detailed endpoint documentation for that category."""


def get_covalent_api_spec() -> str:
    return get_initial_api_info()
