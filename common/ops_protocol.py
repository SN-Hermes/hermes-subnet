from pydantic import BaseModel, Field


class OpsChatCompletionRequest(BaseModel):
    challenge_id: str = Field(default="ops-1", description="Challenge ID associated with the request")
    cid_hash: str = Field(default="", description="CID associated with the request")
    target_uid: int = Field(description="Target miner UID")
    question: str = Field(description="The question to be answered by the miner")
    block_height: int = Field(default=0, description="The block height associated with the request")