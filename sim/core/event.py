from dataclasses import dataclass
from enum import Enum, auto

class EventType(Enum):
    EXTERNAL_ADD = auto()
    EXTERNAL_CANCEL = auto()
    EXTERNAL_MKT = auto()

    SUBMIT = auto()          
    CANCEL = auto()          
    ARRIVE_EXCHANGE = auto() 

    ACK = auto()
    FILL = auto()
    CANCEL_ACK = auto()
    BOOK_TOP = auto()

@dataclass(frozen=True)
class Event:
    ts: int         
    seq: int         
    type: EventType
    payload: dict