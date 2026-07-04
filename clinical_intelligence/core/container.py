from __future__ import annotations
import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Centralized Dependency Injection registry for orchestration interfaces."""

    def __init__(self) -> None:
        self._registry: Dict[Type[Any], Any] = {}
        self._instances: Dict[Type[Any], Any] = {}
        self._is_shutdown = False

    def register_singleton(self, interface: Type[Any], concrete_class: Type[Any], *args: Any, **kwargs: Any) -> None:
        """Register a singleton class mapped to an interface (lazy initialization)."""
        self._registry[interface] = (concrete_class, args, kwargs)
        if interface in self._instances:
            del self._instances[interface]

    def register_instance(self, interface: Type[Any], instance: Any) -> None:
        """Register an already initialized concrete object instance."""
        self._instances[interface] = instance

    def resolve(self, interface: Type[Any]) -> Any:
        """Resolve interface to its concrete instance (instantiating singleton if needed)."""
        if interface in self._instances:
            return self._instances[interface]
        
        if interface in self._registry:
            concrete_class, args, kwargs = self._registry[interface]
            logger.debug(f"ServiceContainer: Lazy initializing {concrete_class.__name__} for {interface.__name__}")
            instance = concrete_class(*args, **kwargs)
            self._instances[interface] = instance
            return instance
            
        raise ValueError(f"ServiceContainer: Interface '{interface.__name__}' is not registered.")

    def health_check(self) -> Dict[str, bool]:
        """Perform health inspections on all resolved services."""
        health_status = {}
        for interface, instance in self._instances.items():
            if hasattr(instance, "health") and callable(instance.health):
                try:
                    health_status[interface.__name__] = bool(instance.health())
                except Exception:
                    health_status[interface.__name__] = False
            else:
                health_status[interface.__name__] = True
        return health_status

    def shutdown(self) -> None:
        """Gracefully release all resources and flush pending tasks."""
        if self._is_shutdown:
            return
        logger.info("ServiceContainer: Initiating graceful shutdown of all services.")
        for interface, instance in list(self._instances.items()):
            if hasattr(instance, "shutdown") and callable(instance.shutdown):
                try:
                    instance.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down service {interface.__name__}: {e}")
        self._instances.clear()
        self._registry.clear()
        self._is_shutdown = True


# Global container registry instance
container = ServiceContainer()
