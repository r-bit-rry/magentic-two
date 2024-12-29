# this file can help ollama reply in json format
# https://ollama.com/blog/structured-outputs

from typing import Literal
from pydantic import BaseModel


class BaseBoolLedgerResponse(BaseModel):
    reason: str
    answer: bool

class InstructionOrQuestion(BaseModel):
    reason: str
    answer: str

class NextSpeaker(BaseModel):
    reason: str
    answer: Literal["FileSurfer", "WebSurfer", "Coder", "Executor", "User"]

class Ledger(BaseModel):
    is_request_satisfied: BaseBoolLedgerResponse
    is_in_loop: BaseBoolLedgerResponse
    is_progress_being_made: BaseBoolLedgerResponse
    next_speaker: NextSpeaker
    instruction_or_question: InstructionOrQuestion
