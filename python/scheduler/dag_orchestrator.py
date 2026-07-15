"""
DAG Orchestrator for Nightly Pipelines.
Sequential execution to prevent memory overlap spikes.
Ensures pipelines run: data -> features -> training -> validation
without exceeding 14GB RAM ceiling.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGTask:
    """Single task in the DAG pipeline."""
    name: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    max_memory_gb: float = 1.0  # Strict memory limit per task
    timeout_sec: float = 3600.0  # 1 hour default
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class DAGOrchestrator:
    """
    Executes tasks in topological order with strict memory bounds.
    Ensures only one heavy task runs at a time to prevent RAM spikes.
    """
    
    MAX_SYSTEM_RAM_GB = 14.0
    RESERVED_RAM_GB = 2.0
    AVAILABLE_RAM_GB = MAX_SYSTEM_RAM_GB - RESERVED_RAM_GB
    
    def __init__(self, sequential_heavy_tasks: bool = True):
        self.tasks: Dict[str, DAGTask] = {}
        self.sequential_heavy = sequential_heavy_tasks
        self.execution_order: List[str] = []
        
    def add_task(
        self,
        name: str,
        func: Callable,
        dependencies: List[str] = None,
        max_memory_gb: float = 1.0,
        timeout_sec: float = 3600.0,
    ):
        """Add a task to the DAG."""
        if dependencies is None:
            dependencies = []
            
        # Validate dependencies exist
        for dep in dependencies:
            if dep not in self.tasks:
                raise ValueError(f"Dependency '{dep}' not found for task '{name}'")
        
        self.tasks[name] = DAGTask(
            name=name,
            func=func,
            dependencies=dependencies,
            max_memory_gb=max_memory_gb,
            timeout_sec=timeout_sec,
        )
        
    def _topological_sort(self) -> List[str]:
        """Compute execution order using Kahn's algorithm."""
        in_degree = {name: len(task.dependencies) for name, task in self.tasks.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            # Sort by memory requirement (heavy tasks last in each batch)
            queue.sort(key=lambda x: self.tasks[x].max_memory_gb)
            node = queue.pop(0)
            order.append(node)
            
            for task_name, task in self.tasks.items():
                if node in task.dependencies:
                    in_degree[task_name] -= 1
                    if in_degree[task_name] == 0:
                        queue.append(task_name)
        
        if len(order) != len(self.tasks):
            raise ValueError("Cycle detected in DAG!")
            
        return order
    
    def _check_memory_available(self, required_gb: float) -> bool:
        """Check if sufficient memory is available."""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        
        # Ensure we stay under the 14GB ceiling
        return available_gb >= (self.RESERVED_RAM_GB + required_gb)
    
    async def _execute_task(self, task: DAGTask) -> bool:
        """Execute a single task with memory monitoring."""
        logger.info(f"Starting task: {task.name}")
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        
        try:
            # Wait for memory to be available
            while not self._check_memory_available(task.max_memory_gb):
                logger.info(f"Waiting for memory... Need {task.max_memory_gb}GB")
                await asyncio.sleep(5.0)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, task.func),
                timeout=task.timeout_sec
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            
            duration = task.end_time - task.start_time
            logger.info(f"Completed task: {task.name} in {duration:.2f}s")
            
            return True
            
        except asyncio.TimeoutError:
            task.error = f"Timeout after {task.timeout_sec}s"
            task.status = TaskStatus.FAILED
            task.end_time = time.time()
            logger.error(f"Task {task.name} timed out")
            return False
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.end_time = time.time()
            logger.error(f"Task {task.name} failed: {e}")
            return False
    
    async def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the entire DAG pipeline."""
        # Compute execution order
        self.execution_order = self._topological_sort()
        logger.info(f"Execution order: {' -> '.join(self.execution_order)}")
        
        results = {}
        failed_tasks = []
        
        for task_name in self.execution_order:
            task = self.tasks[task_name]
            
            # Check if all dependencies completed
            deps_ok = all(
                self.tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.dependencies
            )
            
            if not deps_ok:
                task.status = TaskStatus.SKIPPED
                task.error = "Dependencies failed"
                failed_tasks.append(task_name)
                continue
            
            # Execute task
            success = await self._execute_task(task)
            
            if not success:
                failed_tasks.append(task_name)
                
                # If this is a critical task, abort pipeline
                if task.max_memory_gb > 2.0:  # Heavy task failure
                    logger.critical(f"Critical task {task_name} failed. Aborting pipeline.")
                    break
            
            results[task_name] = {
                "status": task.status.value,
                "duration": (task.end_time - task.start_time) if task.end_time and task.start_time else None,
                "error": task.error,
            }
        
        return {
            "success": len(failed_tasks) == 0,
            "results": results,
            "failed_tasks": failed_tasks,
        }


# Example nightly pipeline
async def run_nightly_pipeline():
    """Example: Data download -> Features -> Training -> Validation"""
    
    orchestrator = DAGOrchestrator(sequential_heavy_tasks=True)
    
    # Task 1: Download market data
    def download_data():
        logger.info("Downloading market data...")
        time.sleep(10)  # Simulate work
        return {"data_path": "/data/market_2024.parquet"}
    
    # Task 2: Feature engineering
    def compute_features():
        logger.info("Computing features...")
        time.sleep(15)
        return {"feature_count": 150}
    
    # Task 3: Model training (heavy)
    def train_model():
        logger.info("Training model...")
        time.sleep(30)
        return {"model_path": "/models/latest.pkl"}
    
    # Task 4: Validation
    def validate_model():
        logger.info("Validating model...")
        time.sleep(10)
        return {"sharpe": 2.5, "drawdown": 0.08}
    
    # Add tasks with dependencies
    orchestrator.add_task("download", download_data, max_memory_gb=0.5)
    orchestrator.add_task("features", compute_features, dependencies=["download"], max_memory_gb=1.0)
    orchestrator.add_task("train", train_model, dependencies=["features"], max_memory_gb=3.0)
    orchestrator.add_task("validate", validate_model, dependencies=["train"], max_memory_gb=1.0)
    
    # Execute
    results = await orchestrator.execute_pipeline()
    
    print("\n=== Pipeline Results ===")
    for task_name, result in results["results"].items():
        status_icon = "✓" if result["status"] == "completed" else "✗"
        print(f"{status_icon} {task_name}: {result['status']} ({result['duration']:.2f}s)")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_nightly_pipeline())
