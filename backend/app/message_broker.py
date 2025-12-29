import logging
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from app.models import NegotiationMessage

logger = logging.getLogger(__name__)

class MessageBroker:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue: List[NegotiationMessage] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def subscribe(
        self,
        department_id: str,
        callback: Callable[[NegotiationMessage], None]
    ):
        
        self.subscribers[department_id].append(callback)
        self.logger.info(f"Subscribed {department_id} to message broker")

    def unsubscribe(self, department_id: str, callback: Callable):
        if department_id in self.subscribers:
            try:
                self.subscribers[department_id].remove(callback)
            except ValueError:
                pass

    def publish(self, message: NegotiationMessage):
        self.message_queue.append(message)

        if message.receiver == "all":

            for dept_id, callbacks in self.subscribers.items():
                for callback in callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        self.logger.error(
                            f"Error in callback for {dept_id}: {e}"
                        )
        elif isinstance(message.receiver, list):

            for dept_id in message.receiver:
                if dept_id in self.subscribers:
                    for callback in self.subscribers[dept_id]:
                        try:
                            callback(message)
                        except Exception as e:
                            self.logger.error(
                                f"Error in callback for {dept_id}: {e}"
                            )
        else:

            if message.receiver in self.subscribers:
                for callback in self.subscribers[message.receiver]:
                    try:
                        callback(message)
                    except Exception as e:
                        self.logger.error(
                            f"Error in callback for {message.receiver}: {e}"
                        )

        self.logger.info(
            f"Published message {message.message_id} from {message.sender}"
        )

    def get_messages(
        self,
        department_id: Optional[str] = None,
        limit: int = 100
    ) -> List[NegotiationMessage]:
        
        messages = self.message_queue

        if department_id:
            messages = [
                msg for msg in messages
                if msg.sender == department_id or (
                    isinstance(msg.receiver, list) and department_id in msg.receiver
                ) or msg.receiver == department_id
            ]

        return messages[-limit:]

    def clear_queue(self):
        self.message_queue = []
