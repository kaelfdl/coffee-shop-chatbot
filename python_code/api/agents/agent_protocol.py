from typing import Protocol, List, Dict, Any, runtime_checkable

@runtime_checkable
class AgentProtocol(Protocol):
    def get_response(self, messages:List[Dict[str, Any]]) -> Dict[str, Any]:
        ...