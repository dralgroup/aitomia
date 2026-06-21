from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages 
from pydantic import BaseModel
from typing import Annotated,List,Optional


class AitomiaState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    messages_to_user: Annotated[List[AnyMessage], add_messages]
    scripts: str = None
    working_directory: str = None # Current working directory
    working_directory_stack: List[str] = None  # The stack of working directories, it is managed by the file manager agent
    result_files: List[str] = None # Messages that contain the result files, e.g., molecules, database, etc.
    current_task_messages: List[List[AnyMessage]] = None
    error: Optional[str] = None  # Error message if any error occurred, used to interrupt the workflow
    has_error: bool = False  # Flag to indicate if an error has occurred
    task_title: Optional[str] = None  # Task title for the conversation
    status: str = None