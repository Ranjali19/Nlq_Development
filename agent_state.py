from pydantic import BaseModel
from typing import Optional, Union, Any
from pandas import DataFrame

class AgentState(BaseModel):
    user_query: str
    clarified_query: Optional[str] = None
    last_ambiguous_query: Optional[str] = None
    is_safe: Optional[bool] = None
    sql_query: Optional[str] = None
    result: Optional[Any] = None
    answer: Optional[str] = None
    needs_clarification: Optional[bool] = None
    clarification_question: Optional[str] = None
    safety_reason: Optional[str] = None
    wants_graph: Optional[bool] = False
    graph: Optional[str] = None
    is_greeting: Optional[bool] = False
    assistant_response: Optional[str] = None
    feedback: Optional[str] = None
    previous_answer: Optional[str] = None
    missing_info: Optional[str] = None
    retries: int = 0  
    waiting_for_clarification: bool = False
    pending_human_review: bool = False
    chat_history: list = []
    model_config = {
        "arbitrary_types_allowed": True
    }
