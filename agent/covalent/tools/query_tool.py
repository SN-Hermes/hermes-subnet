import json
import time
import jq
from typing import Optional, Dict, Union, Any
from pydantic import BaseModel, Field
from loguru import logger

from langchain_core.tools import StructuredTool

from ..context import CovalentContext
from ..service import CovalentService
from ..types import CovalentConfig


MAX_RESPONSE_BYTES = 100 * 1024
JQ_BUILTINS = {
    'add', 'all', 'any', 'del', 'group_by', 'keys', 'length', 'map',
    'max', 'min', 'reverse', 'select', 'sort', 'sort_by', 'tostring',
    'tonumber', 'unique', 'values'
}


class QueryInput(BaseModel):
    path: str = Field(description='REST API path, starting with /v1/')
    params: Optional[Dict[str, Union[str, int, bool]]] = Field(
        default=None,
        description='Query parameters as key-value pairs'
    )


class ResultHeadInput(BaseModel):
    count: int = Field(
        default=20,
        ge=1,
        le=200,
        description='Number of text lines to show from the saved payload'
    )


class ResultJqInput(BaseModel):
    path: str = Field(
        description='jq filter to run against the saved payload (e.g., .items | length, .items[0:20] | map(.contract_ticker_symbol), . for all, or .data.items[0] which will be normalized)'
    )


def create_covalent_query_tool(
    config: CovalentConfig,
    context: CovalentContext,
) -> StructuredTool:
    service = CovalentService(config)

    async def func(path: str, params: Optional[Dict[str, Union[str, int, bool]]] = None) -> str:
        start_time = time.time()
        logger.info(f'Starting Covalent REST request: path={path}, has_params={params is not None}')

        if not path.startswith('/v1/'):
            return '❌ Error: Path must start with /v1/'

        result = await service.execute(path, params)
        execution_time = int((time.time() - start_time) * 1000)
        logger.info(f'Covalent request completed: execution_time={execution_time}ms, has_data={result.data is not None}, has_error={result.error}')

        if result.error:
            logger.error(f'Covalent API error: {result.error_message}')
            return f"❌ API Error: {result.error_message or 'Unknown error'} (code: {result.error_code or 'N/A'})"

        if result.data is not None:
            saved = context.set_result(result.data, path)

            item_count = 'unknown'
            structure_hint = ''

            if isinstance(result.data, list):
                item_count = len(result.data)
                structure_hint = 'array'
            elif isinstance(result.data, dict):
                data_obj = result.data
                if isinstance(data_obj.get('items'), list):
                    item_count = len(data_obj['items'])
                    structure_hint = 'object with items'
                else:
                    array_keys = [k for k, v in data_obj.items() if isinstance(v, list)]
                    if len(array_keys) == 1:
                        array_key = array_keys[0]
                        item_count = len(data_obj[array_key])
                        structure_hint = f'object with {array_key} array'
                    else:
                        item_count = 1
                        structure_hint = 'object'

            logger.info(f'Result cached: id={saved.id}, size_kb={saved.size // 1024}, item_count={item_count}, structure_hint={structure_hint}')

            return f"""✅ Request successful.

📊 Structure: {structure_hint} ({item_count} items)
📦 Size: {saved.size // 1024}KB
⏱️ Time: {execution_time}ms

⚠️ IMPORTANT: Raw Covalent responses use top-level fields named data, error, error_message, and error_code.
👉 This tool caches only the unwrapped payload from data.
👉 Use covalent_result_jq directly when the docs already tell you the shape.
👉 Use covalent_result_head only if jq fails or the payload shape is unclear."""

        logger.warning('Unexpected response format')
        return '⚠️ Unexpected response format'

    return StructuredTool(
        name='covalent_query',
        description="""Execute a REST API request against the Covalent endpoint.

Call covalent_api_info FIRST to understand available endpoints.

Input:
- path: API path starting with /v1/ (e.g., /v1/eth-mainnet/address/0x.../balances_v2/)
- params: Optional query parameters as key-value pairs

Example paths:
- Token balances: /v1/eth-mainnet/address/vitalik.eth/balances_v2/
- Transactions: /v1/eth-mainnet/address/0x.../transactions_v3/
- Token prices: /v1/pricing/historical_by_addresses_v2/eth-mainnet/USD/0x.../
- Token holders: /v1/eth-mainnet/tokens/0x.../token_holders_v2/

Common params:
- quote-currency: USD, EUR, etc.
- page-size: Number of items (default 100)
- page-number: 0-indexed page number
- no-spam: true to filter spam tokens

🛑 CRITICAL RULES:
1. Call covalent_api_info FIRST to understand endpoints
2. Chain names are CASE-SENSITIVE: "eth-mainnet" not "Ethereum"
3. Raw Covalent responses use top-level fields named data, error, error_message, and error_code
4. This tool caches only response.data for downstream inspection
5. After getting results, use covalent_result_jq to extract fields
6. Balance values need division by 10^contract_decimals

Example:
- path: "/v1/eth-mainnet/address/vitalik.eth/balances_v2/"
- params: quote-currency=USD, no-spam=true`,
""",
        func=func,
        args_schema=QueryInput,
        coroutine=func
    )


def create_covalent_result_head_tool(
    context: CovalentContext,
) -> StructuredTool:
    def func(count: int = 20) -> str:
        logger.info(f'covalent_result_head invoked with count={count}')

        result = context.get_result()
        if not result:
            logger.warning('No saved result found')
            return '❌ No saved result found. Run covalent_query first.'

        formatted = result.data if isinstance(result.data, str) else json.dumps(result.data, indent=2)
        lines = formatted.split('\n')
        head_lines = lines[:count]
        preview = '\n'.join(head_lines)
        truncated = len(lines) > count
        large_payload_hint = ('⚠️ Large payload detected. Plan ONE jq call if possible. Avoid repeated full-array scans, especially sort_by/group_by/aggregate passes.\n' 
                             if len(lines) > 5000 else '')

        logger.info(f'covalent_result_head completed: requested={count}, returned={len(head_lines)}, total={len(lines)}, truncated={truncated}')

        return f"""📋 First {len(head_lines)} of {len(lines)} lines from the saved payload:

{preview}

{('... (truncated)\n' if truncated else '')}
{large_payload_hint}
💡 Next: Use covalent_result_jq with a path against the saved payload."""

    return StructuredTool(
        name='covalent_result_head',
        description="""Fallback tool: preview the saved payload as text.

This behaves like a text-based "head" command.
It does not assume arrays, objects, or an items field.
Use it only when covalent_result_jq fails or when the payload shape is still unclear after reading the docs.

Input:
- count: Number of lines to show (default: 20, max: 200)

Output shows:
- The first N lines of the saved payload text
- A truncated preview that helps you choose the jq path
""",
        func=func,
        args_schema=ResultHeadInput
    )


def normalize_data_path(path: str) -> str:
    normalized = path.strip()

    if normalized in ('data', '.data'):
        return '.'

    normalized = normalized.replace('.data', '', 1) if normalized.startswith('.data') else normalized.replace('data', '', 1)
    
    if not normalized or normalized == '.':
        return '.'

    if normalized.startswith('.'):
        return normalized

    if normalized.startswith('['):
        return f'.{normalized}'

    first_token = normalized.split('.')[0].split('[')[0]
    if first_token in JQ_BUILTINS:
        return normalized
    
    return f'.{normalized}'


def get_jq_error_hint(filter_str: str, error_message: str) -> Optional[str]:
    if 'cannot be negated' in error_message:
        return f'A numeric field in the filter is null. Use a default value, for example `{filter_str.replace("-.quote", "-(.quote // 0)")}` or explicitly write `(.quote // 0)`.'
    
    if 'Cannot iterate over null' in error_message:
        return 'Part of the filter is iterating over a null value. Guard it with `// []` for arrays or `// 0` for numbers.'
    
    if 'is not defined' in error_message:
        return 'The filter is referencing a field without jq root access. Use `.items`, `.pagination`, or `.[0]` instead of bare identifiers when needed.'
    
    return None


def run_jq(input_data: Any, filter_str: str) -> Dict[str, Any]:
    """
    Run jq filter using Python jq binding instead of subprocess.
    
    Args:
        input_data: The data to filter (will be converted to JSON if needed)
        filter_str: The jq filter string
        
    Returns:
        Dict with 'success', 'result', and 'error' keys
    """
    try:
        # Compile the jq filter
        compiled_filter = jq.compile(filter_str)
        
        # Apply the filter to the input data
        result = compiled_filter.input(input_data).all()
        
        # jq can return multiple results, we typically want all of them
        if len(result) == 0:
            output = ''
        elif len(result) == 1:
            # Single result - convert to JSON string
            output = json.dumps(result[0], indent=2) if not isinstance(result[0], str) else result[0]
        else:
            # Multiple results - join them
            output = '\n'.join(json.dumps(r, indent=2) if not isinstance(r, str) else r for r in result)
        
        return {
            'success': True,
            'result': output,
            'error': None
        }
    except ValueError as e:
        # jq compilation or execution error
        error_message = str(e)
        return {
            'success': False,
            'result': '',
            'error': error_message
        }
    except Exception as e:
        # Other unexpected errors
        return {
            'success': False,
            'result': '',
            'error': str(e)
        }


def create_covalent_result_jq_tool(
    context: CovalentContext,
) -> StructuredTool:
    async def func(path: str) -> str:
        normalized_path = normalize_data_path(path)
        start_time = time.time()
        logger.info(f'covalent_result_jq invoked: path={path}, normalized={normalized_path}')

        result = context.get_result()
        if not result:
            logger.warning('No saved result found')
            return '❌ No saved result found. Run covalent_query first.'

        try:
            jq_result = run_jq(result.data, normalized_path)
            
            if not jq_result['success']:
                error_message = jq_result['error']
                hint = get_jq_error_hint(normalized_path, error_message)
                logger.error(f'covalent_result_jq failed: {error_message}')
                return f"❌ Failed to extract: {error_message}{(f'\\n\\n💡 Hint: {hint}' if hint else '')}"

            formatted = jq_result['result'].rstrip()
            
            if not formatted:
                return f"""❌ jq filter produced no output: {normalized_path}

Call covalent_result_head again and adjust the filter to match the saved payload."""

            result_size = len(formatted.encode('utf-8'))
            execution_time = int((time.time() - start_time) * 1000)
            
            extracted_type = 'string'
            try:
                parsed = json.loads(formatted)
                extracted_type = f'array[{len(parsed)}]' if isinstance(parsed, list) else type(parsed).__name__
            except:
                pass

            logger.info(f'covalent_result_jq completed: extracted_type={extracted_type}, execution_time={execution_time}ms, size_kb={result_size // 1024}')

            returned_content = formatted
            truncation_notice = ''

            if result_size > MAX_RESPONSE_BYTES:
                while len(returned_content.encode('utf-8')) > MAX_RESPONSE_BYTES:
                    returned_content = returned_content[:-1024]
                
                truncation_notice = f"""⚠️ Output truncated: {result_size // 1024}KB total, returning first {len(returned_content.encode('utf-8')) // 1024}KB.
Use a narrower jq filter such as `.items[0:20]`, `.items | length`, or a specific field projection."""
                
                logger.warning(f'JQ result was truncated: {result_size // 1024}KB → {len(returned_content.encode("utf-8")) // 1024}KB')

            slow_warning = f'⚠️ Slow jq query: {execution_time}ms. Avoid repeated jq calls and prefer one final extraction pass.' if execution_time > 3000 else ''

            return f"""📊 Extracted from {result.id}:

{returned_content}

{(truncation_notice + '\\n\\n') if truncation_notice else ''}{slow_warning}"""

        except Exception as error:
            error_message = str(error)
            hint = get_jq_error_hint(normalized_path, error_message)
            logger.error(f'covalent_result_jq failed: {error_message}')
            return f"❌ Failed to extract: {error_message}{(f'\\n\\n💡 Hint: {hint}' if hint else '')}"

    return StructuredTool(
        name='covalent_result_jq',
        description="""Extract specific fields from the saved result using jq.

Preferred extraction tool after covalent_query.
Different endpoints return different payload structures (array, object with items, etc.).
The raw API response uses top-level fields named data, error, error_message, and error_code,
but this tool operates on the saved payload after unwrapping.
Performance note: large payloads can make jq queries expensive.
Prefer one final jq call instead of several exploratory passes.
Use covalent_result_head only if this jq call fails or the payload shape is unclear.
Keep jq outputs bounded. Avoid extracting hundreds or thousands of values into the model context.

jq examples:
- "." - Return all data
- ".data" - Also returns all data (normalized to ".")
- ".data.items[0]" - Allowed; normalized to ".items[0]"
- ".items[0]" - First item
- ".items[0:5]" - First 5 items
- ".items | length" - Count items
- ".items | map(.quote // 0) | add" - Sum quote values safely when some are null
- ".items[0:20] | map(.contract_ticker_symbol)" - Fast slice to list token symbols
- ".items | sort_by(-(.quote // 0)) | .[0:10] | map(.contract_ticker_symbol)" - More expensive top tokens by quote
- ".[0].prices | map(.price)" - Map nested arrays when the root payload is an array
- Avoid unbounded filters like ".items | map(.contract_ticker_symbol) | unique" on very large payloads

Input:
- path: jq filter based on the saved payload structure""",
        func=func,
        args_schema=ResultJqInput,
        coroutine=func
    )
