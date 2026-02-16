import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.models import ResourcePool, ResourceRequest, ResourceType

logger = logging.getLogger(__name__)

class ResourceManager:
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def create_pool(
        self,
        pool_id: str,
        resource_type: ResourceType,
        total_available: float,
        constraints: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> ResourcePool:
        
        self.logger.info(f"Creating resource pool {pool_id}")

        pool = ResourcePool(
            pool_id=pool_id,
            resource_type=resource_type,
            total_available=total_available,
            constraints=constraints or {},
            description=description
        )

        self.pools[pool_id] = pool
        return pool

    def get_pool(self, pool_id: str) -> Optional[ResourcePool]:
        return self.pools.get(pool_id)

    def list_pools(self) -> List[ResourcePool]:
        return list(self.pools.values())

    def update_pool(
        self,
        pool_id: str,
        total_available: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[ResourcePool]:
        
        pool = self.pools.get(pool_id)
        if not pool:
            return None

        if total_available is not None:
            pool.total_available = total_available

        if constraints is not None:
            pool.constraints.update(constraints)

        pool.updated_at = datetime.now()
        return pool

    def allocate_resources(
        self,
        pool_id: str,
        department_id: str,
        amount: float
    ) -> bool:
        
        pool = self.pools.get(pool_id)
        if not pool:
            self.logger.error(f"Pool {pool_id} not found")
            return False

        if pool.available < amount:
            self.logger.warning(
                f"Insufficient resources in pool {pool_id}. "
                f"Available: {pool.available}, Requested: {amount}"
            )
            return False

        if not self._check_constraints(pool, department_id, amount):
            return False

        current_allocation = pool.allocated.get(department_id, 0)
        pool.allocated[department_id] = current_allocation + amount
        pool.updated_at = datetime.now()

        self.logger.info(
            f"Allocated {amount} to {department_id} from pool {pool_id}"
        )
        return True

    def deallocate_resources(
        self,
        pool_id: str,
        department_id: str,
        amount: float
    ) -> bool:
        
        pool = self.pools.get(pool_id)
        if not pool:
            return False

        current_allocation = pool.allocated.get(department_id, 0)
        if current_allocation < amount:
            self.logger.warning(
                f"Cannot deallocate {amount} from {department_id}. "
                f"Current allocation: {current_allocation}"
            )
            return False

        pool.allocated[department_id] = current_allocation - amount
        if pool.allocated[department_id] == 0:
            del pool.allocated[department_id]

        pool.updated_at = datetime.now()
        return True

    def finalize_allocation(
        self,
        pool_id: str,
        allocations: Dict[str, float]
    ) -> bool:
        
        pool = self.pools.get(pool_id)
        if not pool:
            return False

        total_requested = sum(allocations.values())
        if total_requested > pool.total_available:
            self.logger.error(
                f"Total requested {total_requested} exceeds "
                f"available {pool.total_available}"
            )
            return False

        pool.allocated = {}

        for dept_id, amount in allocations.items():
            pool.allocated[dept_id] = amount

        pool.updated_at = datetime.now()
        self.logger.info(f"Finalized allocations for pool {pool_id}")
        return True

    def add_pending_request(
        self,
        pool_id: str,
        request: ResourceRequest
    ) -> bool:
        
        pool = self.pools.get(pool_id)
        if not pool:
            return False

        pool.pending_requests.append(request)
        pool.updated_at = datetime.now()
        return True

    def clear_pending_requests(self, pool_id: str):
        pool = self.pools.get(pool_id)
        if pool:
            pool.pending_requests = []
            pool.updated_at = datetime.now()

    def _check_constraints(
        self,
        pool: ResourcePool,
        department_id: str,
        amount: float
    ) -> bool:
        
        constraints = pool.constraints

        if "min_allocation" in constraints:
            min_alloc = constraints["min_allocation"]
            if amount < min_alloc:
                self.logger.warning(
                    f"Amount {amount} below minimum {min_alloc}"
                )
                return False

        if "max_allocation" in constraints:
            max_alloc = constraints["max_allocation"]
            if amount > max_alloc:
                self.logger.warning(
                    f"Amount {amount} above maximum {max_alloc}"
                )
                return False

        dept_constraints = constraints.get("department_constraints", {})
        if department_id in dept_constraints:
            dept_max = dept_constraints[department_id].get("max", float('inf'))
            if amount > dept_max:
                return False

        return True
