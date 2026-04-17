workflows_reference = """
# Workflows & Decision Trees

## Universal Response Envelope

Every Covalent API response is wrapped in the same top-level envelope, regardless of endpoint:

```json
{
  "data": { "...": "endpoint-specific payload" },
  "error": false,
  "error_message": null,
  "error_code": null
}
```

### Rules That Apply To Every Endpoint
- The actual endpoint result is always under `.data`
- Always check `.error` before trusting `.data`
- If `.error` is `true`, inspect `.error_message` and `.error_code`
- The three error fields are always `.error`, `.error_message`, and `.error_code`

### Important Agent Note
- `covalent_query` unwraps the HTTP response and saves only `.data` into the result cache
- `covalent_result_head` shows the first text lines of the cached payload and does not assume any object shape
- `covalent_result_jq` operates on the cached payload, not the full envelope
- `covalent_result_jq` supports jq filters against the cached payload, including pipes and aggregation
- Numeric fields such as `quote` may be null on some items; use jq defaults like `(.quote // 0)` when sorting or summing
- Large payloads are expensive to scan repeatedly; prefer one final jq call after `covalent_result_head`

## Quick Decision Tree

**Select the right endpoint based on what the user wants to do:**

### Balance & Portfolio Queries
- Check wallet balance → `getTokenBalancesForWalletAddress` (primary, 1 per call)
- Get historical balance at block → `getHistoricalTokenBalancesForWalletAddress` (specialized, 1 per call)
- Get portfolio value over time → `getHistoricalPortfolioForWalletAddress` (primary, 2 per item)
- Get native token only → `getNativeTokenBalance` (primary, 0.5 per call)

### Transaction Queries
- Get recent transactions → `getRecentTransactionsForAddress` (primary, 0.1 per item)
- Get paginated history → `getTransactionsForAddressV3` (primary, 0.1 per item)
- Get single transaction details → `getTransaction` (primary, 0.1 per call)
- Get ERC20 transfers → `getErc20TransfersForWalletAddress` (primary, 0.05 per item)
- Get transaction summary → `getTransactionSummary` (primary, 1 per call)
"""
