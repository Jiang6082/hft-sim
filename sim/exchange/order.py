from dataclasses import dataclass
from enum import Enum, auto

class Side(Enum):
    BUY = auto()
    SELL = auto()

class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()

class OrderStatus(Enum):
    PENDING = auto()
    LIVE = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELED = auto()
    REJECTED = auto()

@dataclass
class Order:
    oid: int
    owner: str          
    side: Side
    type: OrderType
    price: int | None   # ticks
    qty: int
    filled: int = 0
    status: OrderStatus = OrderStatus.PENDING

    @property
    def remaining(self) -> int:
        return self.qty - self.filled
