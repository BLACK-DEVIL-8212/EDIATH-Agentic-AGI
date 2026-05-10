"""
Scheduler Agent for EDIATH
Advanced scheduling system: cron jobs, reminders, recurring tasks, delays, and time-based triggers
"""

import asyncio
import time
import calendar
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from zoneinfo import ZoneInfo

# Cron parsing
try:
    from croniter import croniter

    CRONITER_AVAILABLE = True
except ImportError:
    CRONITER_AVAILABLE = False

# Human-readable time parsing
try:
    from dateutil.parser import parse as parse_date

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False


class ScheduleType(Enum):
    """Types of schedules"""

    ONCE = "once"  # One-time execution
    INTERVAL = "interval"  # Fixed interval
    CRON = "cron"  # Cron expression
    DELAYED = "delayed"  # Delayed execution
    DAILY = "daily"  # Daily at specific time
    WEEKLY = "weekly"  # Weekly on specific days
    MONTHLY = "monthly"  # Monthly on specific dates
    YEARLY = "yearly"  # Yearly on specific date


class ScheduleStatus(Enum):
    """Schedule status"""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MissedScheduleAction(Enum):
    """Action for missed schedules"""

    SKIP = "skip"  # Skip missed executions
    RUN_IMMEDIATELY = "run_immediately"  # Run as soon as possible
    RUN_NEXT = "run_next"  # Run at next scheduled time


@dataclass
class Schedule:
    """Schedule definition"""

    id: str
    name: str
    schedule_type: ScheduleType
    status: ScheduleStatus
    created_at: datetime
    updated_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    end_date: Optional[datetime] = None

    # Type-specific parameters
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    delay_seconds: Optional[int] = None
    time_of_day: Optional[str] = None  # HH:MM format
    day_of_week: Optional[List[int]] = None  # 0-6 (Monday=0)
    day_of_month: Optional[List[int]] = None  # 1-31
    month_of_year: Optional[List[int]] = None  # 1-12

    # Execution settings
    timezone: str = "UTC"
    missed_action: MissedScheduleAction = MissedScheduleAction.SKIP
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay: int = 60  # seconds

    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """Task to be executed by scheduler"""

    schedule_id: str
    task_id: str
    task_type: str
    task_data: Dict[str, Any]
    created_at: datetime
    execution_count: int = 0
    last_error: Optional[str] = None


class SchedulerAgent:
    """
    Advanced scheduling agent capable of:
    - Cron expression scheduling
    - Interval-based scheduling
    - One-time delayed tasks
    - Daily/weekly/monthly/yearly schedules
    - Reminders and alerts
    - Task persistence (disk/database)
    - Missed schedule handling
    - Retry logic with backoff
    - Concurrent task execution
    - Schedule statistics and monitoring
    - Webhook triggers
    - Callback functions
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Scheduler Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Storage
        self.schedules: Dict[str, Schedule] = {}
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()

        # Persistence
        self.persist_enabled = self.config.get("persist_enabled", True)
        self.data_dir = Path(self.config.get("data_dir", "./scheduler_data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.schedules_file = self.data_dir / "schedules.json"
        self.tasks_file = self.data_dir / "tasks.json"

        # Scheduler state
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.worker_tasks: List[asyncio.Task] = []
        self.num_workers = self.config.get("num_workers", 5)

        # Timezone
        self.default_timezone = ZoneInfo(self.config.get("timezone", "UTC"))

        # Callback registry
        self.callbacks: Dict[str, Callable] = {}

        # Statistics
        self.stats = {
            "total_schedules": 0,
            "active_schedules": 0,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "missed_executions": 0,
            "average_execution_time": 0.0,
        }

        # History
        self.execution_history: List[Dict] = []
        self.max_history = self.config.get("max_history", 1000)

        # Load persisted data
        if self.persist_enabled:
            self._load_data()

        # Start workers
        self._start_workers()

        self.logger.info("Scheduler Agent initialized")

    def _load_data(self):
        """Load schedules and tasks from disk"""
        try:
            if self.schedules_file.exists():
                with open(self.schedules_file, "r") as f:
                    schedules_data = json.load(f)
                    for sched_data in schedules_data:
                        schedule = self._dict_to_schedule(sched_data)
                        self.schedules[schedule.id] = schedule
                self.logger.info(f"Loaded {len(self.schedules)} schedules")

            if self.tasks_file.exists():
                with open(self.tasks_file, "r") as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        task = self._dict_to_task(task_data)
                        self.tasks[task.task_id] = task
                self.logger.info(f"Loaded {len(self.tasks)} tasks")

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")

    def _save_data(self):
        """Save schedules and tasks to disk"""
        if not self.persist_enabled:
            return

        try:
            # Save schedules
            schedules_data = [
                self._schedule_to_dict(s) for s in self.schedules.values()
            ]
            with open(self.schedules_file, "w") as f:
                json.dump(schedules_data, f, indent=2, default=str)

            # Save tasks
            tasks_data = [self._task_to_dict(t) for t in self.tasks.values()]
            with open(self.tasks_file, "w") as f:
                json.dump(tasks_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")

    def _schedule_to_dict(self, schedule: Schedule) -> Dict:
        """Convert Schedule to dictionary"""
        return {
            "id": schedule.id,
            "name": schedule.name,
            "schedule_type": schedule.schedule_type.value,
            "status": schedule.status.value,
            "created_at": schedule.created_at.isoformat(),
            "updated_at": schedule.updated_at.isoformat(),
            "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
            "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
            "run_count": schedule.run_count,
            "max_runs": schedule.max_runs,
            "end_date": schedule.end_date.isoformat() if schedule.end_date else None,
            "cron_expression": schedule.cron_expression,
            "interval_seconds": schedule.interval_seconds,
            "delay_seconds": schedule.delay_seconds,
            "time_of_day": schedule.time_of_day,
            "day_of_week": schedule.day_of_week,
            "day_of_month": schedule.day_of_month,
            "month_of_year": schedule.month_of_year,
            "timezone": schedule.timezone,
            "missed_action": schedule.missed_action.value,
            "retry_on_failure": schedule.retry_on_failure,
            "max_retries": schedule.max_retries,
            "retry_delay": schedule.retry_delay,
            "description": schedule.description,
            "tags": schedule.tags,
            "data": schedule.data,
        }

    def _dict_to_schedule(self, data: Dict) -> Schedule:
        """Convert dictionary to Schedule"""
        return Schedule(
            id=data["id"],
            name=data["name"],
            schedule_type=ScheduleType(data["schedule_type"]),
            status=ScheduleStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_run=(
                datetime.fromisoformat(data["last_run"]) if data["last_run"] else None
            ),
            next_run=(
                datetime.fromisoformat(data["next_run"]) if data["next_run"] else None
            ),
            run_count=data["run_count"],
            max_runs=data["max_runs"],
            end_date=(
                datetime.fromisoformat(data["end_date"]) if data["end_date"] else None
            ),
            cron_expression=data["cron_expression"],
            interval_seconds=data["interval_seconds"],
            delay_seconds=data["delay_seconds"],
            time_of_day=data["time_of_day"],
            day_of_week=data["day_of_week"],
            day_of_month=data["day_of_month"],
            month_of_year=data["month_of_year"],
            timezone=data["timezone"],
            missed_action=MissedScheduleAction(data["missed_action"]),
            retry_on_failure=data["retry_on_failure"],
            max_retries=data["max_retries"],
            retry_delay=data["retry_delay"],
            description=data["description"],
            tags=data["tags"],
            data=data["data"],
        )

    def _task_to_dict(self, task: ScheduledTask) -> Dict:
        """Convert ScheduledTask to dictionary"""
        return {
            "schedule_id": task.schedule_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "task_data": task.task_data,
            "created_at": task.created_at.isoformat(),
            "execution_count": task.execution_count,
            "last_error": task.last_error,
        }

    def _dict_to_task(self, data: Dict) -> ScheduledTask:
        """Convert dictionary to ScheduledTask"""
        return ScheduledTask(
            schedule_id=data["schedule_id"],
            task_id=data["task_id"],
            task_type=data["task_type"],
            task_data=data["task_data"],
            created_at=datetime.fromisoformat(data["created_at"]),
            execution_count=data["execution_count"],
            last_error=data.get("last_error"),
        )

    def _generate_id(self, prefix: str = "sched") -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{timestamp}"

    def _get_current_time(self, timezone: Optional[str] = None) -> datetime:
        """Get current time in specified timezone"""
        tz = ZoneInfo(timezone) if timezone else self.default_timezone
        return datetime.now(tz)

    def _calculate_next_run(
        self, schedule: Schedule, from_time: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Calculate next run time based on schedule type"""
        now = from_time or self._get_current_time(schedule.timezone)

        try:
            if schedule.schedule_type == ScheduleType.ONCE:
                # One-time schedule (no next run after completion)
                return None

            elif schedule.schedule_type == ScheduleType.DELAYED:
                # Delayed execution (only runs once after delay)
                if schedule.last_run is None:
                    # First run: schedule at created_at + delay
                    return schedule.created_at + timedelta(
                        seconds=schedule.delay_seconds
                    )
                else:
                    return None

            elif schedule.schedule_type == ScheduleType.INTERVAL:
                # Fixed interval
                if schedule.last_run:
                    next_time = schedule.last_run + timedelta(
                        seconds=schedule.interval_seconds
                    )
                else:
                    next_time = schedule.created_at

                # Ensure next run is in the future
                while next_time <= now:
                    next_time += timedelta(seconds=schedule.interval_seconds)
                return next_time

            elif schedule.schedule_type == ScheduleType.CRON:
                if not CRONITER_AVAILABLE:
                    self.logger.error("croniter not available for cron scheduling")
                    return None

                base_time = (
                    schedule.last_run if schedule.last_run else schedule.created_at
                )
                cron = croniter(schedule.cron_expression, base_time)
                return cron.get_next(datetime)

            elif schedule.schedule_type == ScheduleType.DAILY:
                return self._calculate_daily_next(schedule, now)

            elif schedule.schedule_type == ScheduleType.WEEKLY:
                return self._calculate_weekly_next(schedule, now)

            elif schedule.schedule_type == ScheduleType.MONTHLY:
                return self._calculate_monthly_next(schedule, now)

            elif schedule.schedule_type == ScheduleType.YEARLY:
                return self._calculate_yearly_next(schedule, now)

            else:
                self.logger.warning(f"Unknown schedule type: {schedule.schedule_type}")
                return None

        except Exception as e:
            self.logger.error(
                f"Failed to calculate next run for {schedule.id}: {str(e)}"
            )
            return None

    def _calculate_daily_next(
        self, schedule: Schedule, now: datetime
    ) -> Optional[datetime]:
        """Calculate next daily run time"""
        if not schedule.time_of_day:
            return None

        # Parse time of day (HH:MM)
        try:
            hour, minute = map(int, schedule.time_of_day.split(":"))
        except:
            return None

        # Calculate next occurrence
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if next_run <= now:
            next_run += timedelta(days=1)

        return next_run

    def _calculate_weekly_next(
        self, schedule: Schedule, now: datetime
    ) -> Optional[datetime]:
        """Calculate next weekly run time"""
        if not schedule.day_of_week or not schedule.time_of_day:
            return None

        # Parse time of day
        try:
            hour, minute = map(int, schedule.time_of_day.split(":"))
        except:
            return None

        # Sort days of week
        days = sorted(schedule.day_of_week)

        # Find next day
        current_weekday = now.weekday()
        next_weekday = None

        for day in days:
            if day >= current_weekday:
                next_weekday = day
                break

        if next_weekday is None:
            next_weekday = days[0]
            days_ahead = (7 - current_weekday) + next_weekday
        else:
            days_ahead = next_weekday - current_weekday

        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if days_ahead > 0:
            next_run += timedelta(days=days_ahead)
        elif days_ahead == 0 and next_run <= now:
            next_run += timedelta(days=7)

        return next_run

    def _calculate_monthly_next(
        self, schedule: Schedule, now: datetime
    ) -> Optional[datetime]:
        """Calculate next monthly run time"""
        if not schedule.day_of_month or not schedule.time_of_day:
            return None

        # Parse time of day
        try:
            hour, minute = map(int, schedule.time_of_day.split(":"))
        except:
            return None

        # Sort days
        days = sorted(schedule.day_of_month)

        # Find next day in current month
        current_day = now.day
        next_day = None

        for day in days:
            if day >= current_day:
                # Check if day exists in current month
                last_day = calendar.monthrange(now.year, now.month)[1]
                if day <= last_day:
                    next_day = day
                    break

        if next_day is None:
            # Move to next month
            if now.month == 12:
                next_year = now.year + 1
                next_month = 1
            else:
                next_year = now.year
                next_month = now.month + 1

            next_day = days[0]
            # Ensure day exists in next month
            last_day = calendar.monthrange(next_year, next_month)[1]
            next_day = min(next_day, last_day)

            next_run = datetime(next_year, next_month, next_day, hour, minute, 0)
            next_run = next_run.replace(tzinfo=now.tzinfo)
        else:
            next_run = now.replace(
                day=next_day, hour=hour, minute=minute, second=0, microsecond=0
            )
            if next_run <= now:
                # Move to next month
                if now.month == 12:
                    next_year = now.year + 1
                    next_month = 1
                else:
                    next_year = now.year
                    next_month = now.month + 1

                last_day = calendar.monthrange(next_year, next_month)[1]
                next_day = min(days[0], last_day)
                next_run = datetime(next_year, next_month, next_day, hour, minute, 0)
                next_run = next_run.replace(tzinfo=now.tzinfo)

        return next_run

    def _calculate_yearly_next(
        self, schedule: Schedule, now: datetime
    ) -> Optional[datetime]:
        """Calculate next yearly run time"""
        if (
            not schedule.month_of_year
            or not schedule.day_of_month
            or not schedule.time_of_day
        ):
            return None

        # Parse time of day
        try:
            hour, minute = map(int, schedule.time_of_day.split(":"))
        except:
            return None

        # Sort months
        months = sorted(schedule.month_of_year)

        # Find next month
        current_month = now.month
        next_month = None
        next_year = now.year

        for month in months:
            if month >= current_month:
                next_month = month
                break

        if next_month is None:
            next_month = months[0]
            next_year += 1

        # Check if day exists
        last_day = calendar.monthrange(next_year, next_month)[1]
        next_day = min(schedule.day_of_month[0], last_day)

        next_run = datetime(next_year, next_month, next_day, hour, minute, 0)
        next_run = next_run.replace(tzinfo=now.tzinfo)

        if next_run <= now:
            next_run = next_run.replace(year=next_run.year + 1)

        return next_run

    async def add_schedule(
        self, name: str, schedule_type: ScheduleType, **kwargs
    ) -> Dict[str, Any]:
        """
        Add a new schedule

        Args:
            name: Schedule name
            schedule_type: Type of schedule
            **kwargs: Type-specific parameters

        Returns:
            Dictionary with schedule creation result
        """
        schedule_id = self._generate_id()
        now = self._get_current_time()

        # Create schedule
        schedule = Schedule(
            id=schedule_id,
            name=name,
            schedule_type=schedule_type,
            status=ScheduleStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            **kwargs,
        )

        # Calculate first next run
        schedule.next_run = self._calculate_next_run(schedule)

        if schedule.next_run is None and schedule_type != ScheduleType.ONCE:
            return {
                "success": False,
                "error": f"Failed to calculate next run for schedule type {schedule_type.value}",
            }

        # Store schedule
        self.schedules[schedule_id] = schedule

        # Save to disk
        if self.persist_enabled:
            self._save_data()

        # Update statistics
        self.stats["total_schedules"] += 1
        self.stats["active_schedules"] += 1

        self.logger.info(f"Added schedule: {name} (ID: {schedule_id})")

        return {
            "success": True,
            "schedule_id": schedule_id,
            "name": name,
            "schedule_type": schedule_type.value,
            "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
            "message": f"Schedule {name} created successfully",
        }

    async def add_cron_schedule(
        self, name: str, cron_expression: str, task_data: Dict, **kwargs
    ) -> Dict[str, Any]:
        """
        Add a cron-based schedule

        Args:
            name: Schedule name
            cron_expression: Cron expression (e.g., "0 9 * * *")
            task_data: Task data to execute
            **kwargs: Additional schedule parameters

        Returns:
            Dictionary with schedule creation result
        """
        return await self.add_schedule(
            name=name,
            schedule_type=ScheduleType.CRON,
            cron_expression=cron_expression,
            data=task_data,
            **kwargs,
        )

    async def add_interval_schedule(
        self, name: str, interval_seconds: int, task_data: Dict, **kwargs
    ) -> Dict[str, Any]:
        """
        Add an interval-based schedule

        Args:
            name: Schedule name
            interval_seconds: Interval in seconds
            task_data: Task data to execute
            **kwargs: Additional schedule parameters

        Returns:
            Dictionary with schedule creation result
        """
        return await self.add_schedule(
            name=name,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=interval_seconds,
            data=task_data,
            **kwargs,
        )

    async def add_daily_schedule(
        self, name: str, time_of_day: str, task_data: Dict, **kwargs
    ) -> Dict[str, Any]:
        """
        Add a daily schedule

        Args:
            name: Schedule name
            time_of_day: Time of day (HH:MM format)
            task_data: Task data to execute
            **kwargs: Additional schedule parameters

        Returns:
            Dictionary with schedule creation result
        """
        return await self.add_schedule(
            name=name,
            schedule_type=ScheduleType.DAILY,
            time_of_day=time_of_day,
            data=task_data,
            **kwargs,
        )

    async def add_weekly_schedule(
        self,
        name: str,
        day_of_week: List[int],
        time_of_day: str,
        task_data: Dict,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add a weekly schedule

        Args:
            name: Schedule name
            day_of_week: Days of week (0=Monday, 6=Sunday)
            time_of_day: Time of day (HH:MM format)
            task_data: Task data to execute
            **kwargs: Additional schedule parameters

        Returns:
            Dictionary with schedule creation result
        """
        return await self.add_schedule(
            name=name,
            schedule_type=ScheduleType.WEEKLY,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
            data=task_data,
            **kwargs,
        )

    async def add_once_schedule(
        self, name: str, run_at: datetime, task_data: Dict, **kwargs
    ) -> Dict[str, Any]:
        """
        Add a one-time schedule

        Args:
            name: Schedule name
            run_at: When to run
            task_data: Task data to execute
            **kwargs: Additional schedule parameters

        Returns:
            Dictionary with schedule creation result
        """
        # Create schedule that runs once at specified time
        schedule_id = self._generate_id()
        now = self._get_current_time()

        schedule = Schedule(
            id=schedule_id,
            name=name,
            schedule_type=ScheduleType.ONCE,
            status=ScheduleStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            next_run=run_at,
            data=task_data,
            **kwargs,
        )

        self.schedules[schedule_id] = schedule

        if self.persist_enabled:
            self._save_data()

        self.stats["total_schedules"] += 1
        self.stats["active_schedules"] += 1

        return {
            "success": True,
            "schedule_id": schedule_id,
            "name": name,
            "run_at": run_at.isoformat(),
            "message": f"One-time schedule {name} created",
        }

    async def add_delayed_schedule(
        self, name: str, delay_seconds: int, task_data: Dict, **kwargs
    ) -> Dict[str, Any]:
        """
        Add a delayed schedule (runs once after delay)

        Args:
            name: Schedule name
            delay_seconds: Delay in seconds
            task_data: Task data to execute
            **kwargs: Additional schedule parameters

        Returns:
            Dictionary with schedule creation result
        """
        return await self.add_schedule(
            name=name,
            schedule_type=ScheduleType.DELAYED,
            delay_seconds=delay_seconds,
            data=task_data,
            **kwargs,
        )

    async def remove_schedule(self, schedule_id: str) -> Dict[str, Any]:
        """
        Remove a schedule

        Args:
            schedule_id: ID of schedule to remove

        Returns:
            Dictionary with removal result
        """
        if schedule_id not in self.schedules:
            return {"success": False, "error": f"Schedule {schedule_id} not found"}

        schedule = self.schedules[schedule_id]
        name = schedule.name

        del self.schedules[schedule_id]

        if self.persist_enabled:
            self._save_data()

        self.stats["active_schedules"] -= 1

        self.logger.info(f"Removed schedule: {name} (ID: {schedule_id})")

        return {
            "success": True,
            "schedule_id": schedule_id,
            "message": f"Schedule {name} removed",
        }

    async def pause_schedule(self, schedule_id: str) -> Dict[str, Any]:
        """
        Pause a schedule

        Args:
            schedule_id: ID of schedule to pause

        Returns:
            Dictionary with pause result
        """
        if schedule_id not in self.schedules:
            return {"success": False, "error": f"Schedule {schedule_id} not found"}

        schedule = self.schedules[schedule_id]
        schedule.status = ScheduleStatus.PAUSED
        schedule.updated_at = self._get_current_time()

        if self.persist_enabled:
            self._save_data()

        self.stats["active_schedules"] -= 1

        return {
            "success": True,
            "schedule_id": schedule_id,
            "message": f"Schedule {schedule.name} paused",
        }

    async def resume_schedule(self, schedule_id: str) -> Dict[str, Any]:
        """
        Resume a paused schedule

        Args:
            schedule_id: ID of schedule to resume

        Returns:
            Dictionary with resume result
        """
        if schedule_id not in self.schedules:
            return {"success": False, "error": f"Schedule {schedule_id} not found"}

        schedule = self.schedules[schedule_id]
        schedule.status = ScheduleStatus.ACTIVE
        schedule.updated_at = self._get_current_time()

        # Recalculate next run
        schedule.next_run = self._calculate_next_run(schedule)

        if self.persist_enabled:
            self._save_data()

        self.stats["active_schedules"] += 1

        return {
            "success": True,
            "schedule_id": schedule_id,
            "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
            "message": f"Schedule {schedule.name} resumed",
        }

    async def register_callback(
        self, task_type: str, callback: Callable
    ) -> Dict[str, Any]:
        """
        Register a callback function for a task type

        Args:
            task_type: Type of task (e.g., 'webhook', 'function')
            callback: Async function to call when task executes

        Returns:
            Dictionary with registration result
        """
        self.callbacks[task_type] = callback

        return {
            "success": True,
            "task_type": task_type,
            "message": f"Callback registered for {task_type}",
        }

    async def _execute_task(self, schedule: Schedule, task: ScheduledTask):
        """Execute a scheduled task"""
        start_time = time.time()

        try:
            self.logger.info(
                f"Executing task for schedule: {schedule.name} (ID: {schedule.id})"
            )

            # Execute based on task type
            task_type = task.task_type
            task_data = task.task_data

            if task_type == "function" and task_type in self.callbacks:
                # Execute registered callback
                callback = self.callbacks[task_type]
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(schedule, task_data)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, callback, schedule, task_data
                    )

                success = (
                    result.get("success", True) if isinstance(result, dict) else True
                )

            elif task_type == "webhook":
                # Execute webhook
                success = await self._execute_webhook(task_data)

            elif task_type == "notification":
                # Send notification
                success = await self._execute_notification(task_data)

            else:
                self.logger.warning(f"Unknown task type: {task_type}")
                success = False

            execution_time = time.time() - start_time

            if success:
                self.stats["successful_executions"] += 1
                self.logger.info(
                    f"Task completed successfully in {execution_time:.2f}s"
                )
            else:
                self.stats["failed_executions"] += 1
                self.logger.warning("Task failed")

            # Update statistics
            total_exec = self.stats["total_executions"]
            avg_time = self.stats["average_execution_time"]
            self.stats["average_execution_time"] = (
                avg_time * (total_exec - 1) + execution_time
            ) / total_exec

            # Add to history
            self._add_to_history(
                {
                    "schedule_id": schedule.id,
                    "schedule_name": schedule.name,
                    "task_type": task_type,
                    "success": success,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            self.logger.error(f"Task execution error: {str(e)}")
            self.stats["failed_executions"] += 1
            task.last_error = str(e)

    async def _execute_webhook(self, task_data: Dict) -> bool:
        """Execute webhook task"""
        import aiohttp

        url = task_data.get("url")
        method = task_data.get("method", "POST")
        headers = task_data.get("headers", {})
        body = task_data.get("body", {})

        if not url:
            self.logger.error("Webhook URL not provided")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        return response.status < 400
                else:
                    async with session.post(
                        url, json=body, headers=headers
                    ) as response:
                        return response.status < 400

        except Exception as e:
            self.logger.error(f"Webhook execution error: {str(e)}")
            return False

    async def _execute_notification(self, task_data: Dict) -> bool:
        """Execute notification task"""
        # This would integrate with NotificationAgent
        # For now, just log
        self.logger.info(f"Notification: {task_data.get('message', 'No message')}")
        return True

    async def _process_schedules(self):
        """Main scheduler loop - check and execute due schedules"""
        while self.running:
            try:
                now = self._get_current_time()

                # Check each active schedule
                for schedule_id, schedule in list(self.schedules.items()):
                    if schedule.status != ScheduleStatus.ACTIVE:
                        continue

                    if schedule.next_run and schedule.next_run <= now:
                        # Schedule is due
                        self.stats["total_executions"] += 1

                        # Check if missed
                        if (
                            now - schedule.next_run
                        ).total_seconds() > 60:  # More than 1 minute late
                            self.stats["missed_executions"] += 1

                            if schedule.missed_action == MissedScheduleAction.SKIP:
                                self.logger.info(
                                    f"Skipping missed schedule: {schedule.name}"
                                )
                                # Update next run without executing
                                schedule.last_run = schedule.next_run
                                schedule.next_run = self._calculate_next_run(schedule)
                                schedule.updated_at = now
                                continue

                            elif (
                                schedule.missed_action
                                == MissedScheduleAction.RUN_IMMEDIATELY
                            ):
                                self.logger.info(
                                    f"Running missed schedule immediately: {schedule.name}"
                                )
                                # Execute now

                        # Create task
                        task = ScheduledTask(
                            schedule_id=schedule.id,
                            task_id=self._generate_id("task"),
                            task_type="function",  # Default
                            task_data=schedule.data,
                            created_at=now,
                        )

                        # Execute task
                        await self._execute_task(schedule, task)

                        # Update schedule
                        schedule.last_run = schedule.next_run
                        schedule.run_count += 1
                        schedule.next_run = self._calculate_next_run(schedule)
                        schedule.updated_at = now

                        # Check if schedule should be completed
                        if (
                            schedule.max_runs
                            and schedule.run_count >= schedule.max_runs
                        ):
                            schedule.status = ScheduleStatus.COMPLETED
                            self.stats["active_schedules"] -= 1
                            self.logger.info(
                                f"Schedule {schedule.name} completed (max runs reached)"
                            )

                        if (
                            schedule.end_date
                            and schedule.next_run
                            and schedule.next_run > schedule.end_date
                        ):
                            schedule.status = ScheduleStatus.COMPLETED
                            self.stats["active_schedules"] -= 1
                            self.logger.info(
                                f"Schedule {schedule.name} completed (end date reached)"
                            )

                        # Save changes
                        if self.persist_enabled:
                            self._save_data()

                # Sleep for a short interval
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Scheduler loop error: {str(e)}")
                await asyncio.sleep(5)

    def _start_workers(self):
        """Start worker tasks"""
        self.running = True
        self.scheduler_task = asyncio.create_task(self._process_schedules())
        self.logger.info("Scheduler started")

    async def _stop_workers(self):
        """Stop worker tasks"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        for task in self.worker_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.logger.info("Scheduler stopped")

    def _add_to_history(self, entry: Dict):
        """Add entry to execution history"""
        self.execution_history.append(entry)
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history :]

    async def get_schedule(self, schedule_id: str) -> Dict[str, Any]:
        """Get schedule details"""
        if schedule_id not in self.schedules:
            return {"success": False, "error": f"Schedule {schedule_id} not found"}

        schedule = self.schedules[schedule_id]

        return {"success": True, "schedule": self._schedule_to_dict(schedule)}

    async def get_all_schedules(self, status: Optional[str] = None) -> Dict[str, Any]:
        """Get all schedules"""
        schedules = list(self.schedules.values())

        if status:
            schedules = [s for s in schedules if s.status.value == status]

        return {
            "success": True,
            "total": len(schedules),
            "schedules": [self._schedule_to_dict(s) for s in schedules],
        }

    async def get_upcoming_runs(self, limit: int = 10) -> Dict[str, Any]:
        """Get upcoming schedule runs"""
        upcoming = []

        for schedule in self.schedules.values():
            if schedule.status == ScheduleStatus.ACTIVE and schedule.next_run:
                upcoming.append(
                    {
                        "schedule_id": schedule.id,
                        "name": schedule.name,
                        "next_run": schedule.next_run.isoformat(),
                        "schedule_type": schedule.schedule_type.value,
                    }
                )

        # Sort by next run time
        upcoming.sort(key=lambda x: x["next_run"])

        return {"success": True, "upcoming": upcoming[:limit], "total": len(upcoming)}

    def get_history(self, limit: int = None, success_only: bool = False) -> List[Dict]:
        """Get execution history"""
        history = self.execution_history

        if success_only:
            history = [h for h in history if h["success"]]

        if limit:
            history = history[-limit:]

        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "history_size": len(self.execution_history),
            "schedules_by_type": self._count_by_type(),
            "schedules_by_status": self._count_by_status(),
            "is_running": self.running,
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count schedules by type"""
        counts = {}
        for schedule in self.schedules.values():
            type_name = schedule.schedule_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def _count_by_status(self) -> Dict[str, int]:
        """Count schedules by status"""
        counts = {}
        for schedule in self.schedules.values():
            status_name = schedule.status.value
            counts[status_name] = counts.get(status_name, 0) + 1
        return counts

    async def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        self.logger.info("Execution history cleared")

    async def shutdown(self):
        """Gracefully shutdown the scheduler"""
        self.logger.info("Shutting down scheduler...")
        await self._stop_workers()

        if self.persist_enabled:
            self._save_data()

        self.logger.info("Scheduler shutdown complete")


# Integration wrapper for EDIATH
class SchedulerAgentWrapper:
    """
    Wrapper class to integrate SchedulerAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.scheduler_agent = SchedulerAgent(config)
        self.agent_type = "scheduler"
        self.capabilities = [
            "add_schedule",
            "remove_schedule",
            "pause_resume",
            "cron_scheduling",
            "interval_scheduling",
            "reminders",
            "webhook_triggers",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a scheduler request

        Request format:
        {
            'operation': 'add|remove|pause|resume|list|upcoming|history',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "add":
            schedule_type = ScheduleType(request.get("schedule_type"))

            # Build kwargs based on schedule type
            kwargs = {
                "name": request.get("name"),
                "schedule_type": schedule_type,
                "task_data": request.get("task_data", {}),
                "description": request.get("description"),
                "tags": request.get("tags", []),
                "max_runs": request.get("max_runs"),
                "end_date": request.get("end_date"),
                "timezone": request.get("timezone", "UTC"),
                "missed_action": MissedScheduleAction(
                    request.get("missed_action", "skip")
                ),
            }

            # Add type-specific parameters
            if schedule_type == ScheduleType.CRON:
                kwargs["cron_expression"] = request.get("cron_expression")
            elif schedule_type == ScheduleType.INTERVAL:
                kwargs["interval_seconds"] = request.get("interval_seconds")
            elif schedule_type == ScheduleType.DAILY:
                kwargs["time_of_day"] = request.get("time_of_day")
            elif schedule_type == ScheduleType.WEEKLY:
                kwargs["day_of_week"] = request.get("day_of_week")
                kwargs["time_of_day"] = request.get("time_of_day")
            elif schedule_type == ScheduleType.MONTHLY:
                kwargs["day_of_month"] = request.get("day_of_month")
                kwargs["time_of_day"] = request.get("time_of_day")
            elif schedule_type == ScheduleType.DELAYED:
                kwargs["delay_seconds"] = request.get("delay_seconds")

            return await self.scheduler_agent.add_schedule(**kwargs)

        elif operation == "remove":
            return await self.scheduler_agent.remove_schedule(
                schedule_id=request.get("schedule_id")
            )

        elif operation == "pause":
            return await self.scheduler_agent.pause_schedule(
                schedule_id=request.get("schedule_id")
            )

        elif operation == "resume":
            return await self.scheduler_agent.resume_schedule(
                schedule_id=request.get("schedule_id")
            )

        elif operation == "list":
            return await self.scheduler_agent.get_all_schedules(
                status=request.get("status")
            )

        elif operation == "get":
            return await self.scheduler_agent.get_schedule(
                schedule_id=request.get("schedule_id")
            )

        elif operation == "upcoming":
            return await self.scheduler_agent.get_upcoming_runs(
                limit=request.get("limit", 10)
            )

        elif operation == "history":
            return {
                "success": True,
                "history": self.scheduler_agent.get_history(
                    limit=request.get("limit"),
                    success_only=request.get("success_only", False),
                ),
            }

        elif operation == "stats":
            return self.scheduler_agent.get_stats()

        elif operation == "clear_history":
            await self.scheduler_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        elif operation == "shutdown":
            await self.scheduler_agent.shutdown()
            return {"success": True, "message": "Scheduler shutdown"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "SchedulerAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.scheduler_agent.get_stats(),
            "schedule_types": [t.value for t in ScheduleType],
        }

    async def close(self):
        """Clean up resources"""
        await self.scheduler_agent.shutdown()


# Example usage and testing
async def test_scheduler_agent():
    """Test the scheduler agent functionality"""

    # Initialize agent
    agent = SchedulerAgent()

    print("=== Scheduler Agent Test ===\n")

    # Test adding a delayed schedule
    print("1. Adding Delayed Schedule...")
    result = await agent.add_delayed_schedule(
        name="Test Delay",
        delay_seconds=3,
        task_data={"message": "This is a delayed test"},
    )
    print(f"   Schedule created: {result['success']}")
    if result["success"]:
        print(f"   Next run: {result.get('next_run', 'N/A')}")

    # Test adding a daily schedule
    print("\n2. Adding Daily Schedule...")
    result = await agent.add_daily_schedule(
        name="Daily Report",
        time_of_day="09:00",
        task_data={"type": "report", "name": "daily_stats"},
    )
    print(f"   Daily schedule: {result['success']}")

    # Test adding a cron schedule
    print("\n3. Adding Cron Schedule...")
    result = await agent.add_cron_schedule(
        name="Hourly Job",
        cron_expression="0 * * * *",
        task_data={"type": "webhook", "url": "https://example.com/hourly"},
    )
    print(f"   Cron schedule: {result['success']}")
    if result["success"]:
        print(f"   Next run: {result.get('next_run', 'N/A')}")

    # Test adding interval schedule
    print("\n4. Adding Interval Schedule...")
    result = await agent.add_interval_schedule(
        name="Every 10 seconds",
        interval_seconds=10,
        task_data={"message": "Interval test"},
        max_runs=3,  # Run only 3 times
    )
    print(f"   Interval schedule: {result['success']}")

    # Test adding weekly schedule
    print("\n5. Adding Weekly Schedule...")
    result = await agent.add_weekly_schedule(
        name="Weekly Meeting",
        day_of_week=[0],  # Monday
        time_of_day="14:30",
        task_data={"type": "reminder", "message": "Team meeting"},
    )
    print(f"   Weekly schedule: {result['success']}")

    # Get all schedules
    print("\n6. All Schedules...")
    result = await agent.get_all_schedules()
    print(f"   Total schedules: {result['total']}")
    for sched in result["schedules"]:
        print(
            f"     - {sched['name']} ({sched['schedule_type']}) - Status: {sched['status']}"
        )

    # Get upcoming runs
    print("\n7. Upcoming Runs...")
    result = await agent.get_upcoming_runs(limit=5)
    print(f"   Total upcoming: {result['total']}")
    for run in result["upcoming"][:3]:
        print(f"     - {run['name']}: {run['next_run']}")

    # Wait for delayed schedule to execute
    print("\n8. Waiting for delayed schedule execution...")
    await asyncio.sleep(5)

    # Get statistics
    print("\n9. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total schedules: {stats['total_schedules']}")
    print(f"   Active schedules: {stats['active_schedules']}")
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   Successful: {stats['successful_executions']}")
    print(f"   Failed: {stats['failed_executions']}")
    print(f"   Average execution time: {stats['average_execution_time']:.3f}s")

    # Get execution history
    print("\n10. Execution History...")
    history = agent.get_history(limit=5)
    for entry in history:
        print(
            f"     {entry['schedule_name']}: {'✓' if entry['success'] else '✗'} ({entry['execution_time']:.2f}s)"
        )

    # Clean up - remove test schedules
    print("\n11. Cleaning up...")
    result = await agent.get_all_schedules()
    for sched in result["schedules"]:
        if "Test" in sched["name"] or "Every" in sched["name"]:
            await agent.remove_schedule(sched["id"])
            print(f"     Removed: {sched['name']}")

    # Shutdown
    await agent.shutdown()

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_scheduler_agent())
