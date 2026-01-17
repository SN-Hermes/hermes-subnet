from enum import Enum


class ErrorCode(Enum):
    """
    error code enum
    """
    SUCCESS = 200
    
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    REQUEST_TIMEOUT = 408
    TOO_MANY_REQUESTS = 429
    
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
    
    TOOL_ERROR = 1001
    LLM_ERROR = 1002

    ## ============ miner side error ============
    AGENT_NOT_FOUND = 2001
    NOT_HEALTHY = 2002

    ## ============ validator side error ============
    FORWARD_SYNTHETIC_FAILED = 3001
    ORGANIC_NO_AVAILABLE_MINERS = 3002
    ORGANIC_NO_SELECTED_MINER = 3003
    ORGANIC_NO_AXON = 3004
    ORGANIC_ERROR_RESPONSE = 3005

class ChallengeType(Enum):
    SYNTHETIC = 1
    ORGANIC_STREAM = 2
    ORGANIC_NONSTREAM = 3

class RoleFlag(Enum):
    NONE = 0
    MINER = 1
    VALIDATOR = 2

