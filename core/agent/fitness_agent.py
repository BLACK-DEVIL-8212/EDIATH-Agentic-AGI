"""
Fitness Agent for EDIATH
Advanced fitness tracking: workouts, nutrition, goals, progress tracking, activity monitoring
"""

import asyncio
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Optional: For BMI calculations and health metrics
try:
    import numpy as np

    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False


class ActivityType(Enum):
    """Types of fitness activities"""

    RUNNING = "running"
    WALKING = "walking"
    CYCLING = "cycling"
    SWIMMING = "swimming"
    WEIGHTLIFTING = "weightlifting"
    YOGA = "yoga"
    HIIT = "hiit"
    CARDIO = "cardio"
    STRETCHING = "stretching"
    HIKING = "hiking"
    DANCING = "dancing"
    SPORTS = "sports"
    CUSTOM = "custom"


class IntensityLevel(Enum):
    """Exercise intensity levels"""

    VERY_LOW = "very_low"  # 1-2 RPE
    LOW = "low"  # 2-3 RPE
    MODERATE = "moderate"  # 4-6 RPE
    HIGH = "high"  # 7-8 RPE
    VERY_HIGH = "very_high"  # 9-10 RPE


class GoalType(Enum):
    """Types of fitness goals"""

    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    ENDURANCE = "endurance"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"
    GENERAL_FITNESS = "general_fitness"
    CUSTOM = "custom"


class NutritionGoal(Enum):
    """Nutrition goals"""

    WEIGHT_LOSS = "weight_loss"
    WEIGHT_MAINTENANCE = "maintenance"
    MUSCLE_GAIN = "muscle_gain"
    PERFORMANCE = "performance"
    HEALTH = "health"


@dataclass
class Workout:
    """Workout session"""

    id: str
    activity_type: ActivityType
    date: date
    duration_minutes: int
    intensity: IntensityLevel
    calories_burned: Optional[int] = None
    distance_km: Optional[float] = None
    heart_rate_avg: Optional[int] = None
    heart_rate_max: Optional[int] = None
    notes: str = ""
    location: str = ""
    completed: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class NutritionLog:
    """Nutrition entry"""

    id: str
    date: date
    meal_type: str  # breakfast, lunch, dinner, snack
    food_name: str
    calories: int
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    fiber_g: float = 0
    sugar_g: float = 0
    sodium_mg: int = 0
    water_ml: int = 0
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BodyMetric:
    """Body measurements"""

    id: str
    date: date
    weight_kg: Optional[float] = None
    body_fat_percent: Optional[float] = None
    muscle_mass_kg: Optional[float] = None
    bmi: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    chest_cm: Optional[float] = None
    arm_cm: Optional[float] = None
    thigh_cm: Optional[float] = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FitnessGoal:
    """Fitness goal"""

    id: str
    type: GoalType
    target_value: float
    current_value: float
    unit: str
    start_date: date
    target_date: date
    progress: float = 0.0
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DailySummary:
    """Daily fitness summary"""

    date: date
    total_calories_burned: int
    total_calories_consumed: int
    net_calories: int
    total_workout_minutes: int
    workouts_count: int
    steps: Optional[int] = None
    sleep_hours: Optional[float] = None
    water_ml: int = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0


class FitnessAgent:
    """
    Advanced fitness tracking agent capable of:
    - Workout logging and tracking
    - Nutrition logging and analysis
    - Body metrics tracking
    - Goal setting and progress monitoring
    - Calorie calculations
    - Fitness recommendations
    - Progress reports and analytics
    - Workout plans generation
    - Health metrics calculations (BMI, BMR, TDEE)
    - Activity streak tracking
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Fitness Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # User profile
        self.user_profile = self.config.get(
            "user_profile",
            {
                "age": 30,
                "gender": "male",
                "height_cm": 170,
                "weight_kg": 70,
                "activity_level": "moderate",  # sedentary, light, moderate, active, very_active
                "fitness_level": "intermediate",
            },
        )

        # Storage
        self.data_dir = Path(self.config.get("data_dir", "./fitness_data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.workouts: List[Workout] = []
        self.nutrition_logs: List[NutritionLog] = []
        self.body_metrics: List[BodyMetric] = []
        self.goals: List[FitnessGoal] = []

        # Statistics
        self.stats = {
            "total_workouts": 0,
            "total_calories_burned": 0,
            "total_workout_minutes": 0,
            "current_streak_days": 0,
            "longest_streak_days": 0,
            "goals_completed": 0,
        }

        # Activity constants
        self.met_values = {
            ActivityType.RUNNING: 9.8,
            ActivityType.WALKING: 3.5,
            ActivityType.CYCLING: 8.0,
            ActivityType.SWIMMING: 8.0,
            ActivityType.WEIGHTLIFTING: 6.0,
            ActivityType.YOGA: 3.0,
            ActivityType.HIIT: 8.5,
            ActivityType.CARDIO: 7.0,
            ActivityType.STRETCHING: 2.5,
            ActivityType.HIKING: 6.0,
            ActivityType.DANCING: 5.5,
            ActivityType.SPORTS: 7.5,
        }

        # Load data
        self._load_data()

        # Calculate current streak
        self._update_streak()

        self.logger.info("Fitness Agent initialized")

    def _load_data(self):
        """Load fitness data from disk"""
        try:
            # Load workouts
            workouts_file = self.data_dir / "workouts.json"
            if workouts_file.exists():
                with open(workouts_file, "r") as f:
                    data = json.load(f)
                    self.workouts = [self._dict_to_workout(w) for w in data]

            # Load nutrition logs
            nutrition_file = self.data_dir / "nutrition.json"
            if nutrition_file.exists():
                with open(nutrition_file, "r") as f:
                    data = json.load(f)
                    self.nutrition_logs = [self._dict_to_nutrition(n) for n in data]

            # Load body metrics
            metrics_file = self.data_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    self.body_metrics = [self._dict_to_metric(m) for m in data]

            # Load goals
            goals_file = self.data_dir / "goals.json"
            if goals_file.exists():
                with open(goals_file, "r") as f:
                    data = json.load(f)
                    self.goals = [self._dict_to_goal(g) for g in data]

            self._update_stats()

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")

    def _save_data(self):
        """Save fitness data to disk"""
        try:
            # Save workouts
            with open(self.data_dir / "workouts.json", "w") as f:
                json.dump(
                    [self._workout_to_dict(w) for w in self.workouts],
                    f,
                    indent=2,
                    default=str,
                )

            # Save nutrition logs
            with open(self.data_dir / "nutrition.json", "w") as f:
                json.dump(
                    [self._nutrition_to_dict(n) for n in self.nutrition_logs],
                    f,
                    indent=2,
                    default=str,
                )

            # Save body metrics
            with open(self.data_dir / "metrics.json", "w") as f:
                json.dump(
                    [self._metric_to_dict(m) for m in self.body_metrics],
                    f,
                    indent=2,
                    default=str,
                )

            # Save goals
            with open(self.data_dir / "goals.json", "w") as f:
                json.dump(
                    [self._goal_to_dict(g) for g in self.goals],
                    f,
                    indent=2,
                    default=str,
                )

        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")

    def _workout_to_dict(self, workout: Workout) -> Dict:
        """Convert Workout to dictionary"""
        return {
            "id": workout.id,
            "activity_type": workout.activity_type.value,
            "date": workout.date.isoformat(),
            "duration_minutes": workout.duration_minutes,
            "intensity": workout.intensity.value,
            "calories_burned": workout.calories_burned,
            "distance_km": workout.distance_km,
            "heart_rate_avg": workout.heart_rate_avg,
            "heart_rate_max": workout.heart_rate_max,
            "notes": workout.notes,
            "location": workout.location,
            "completed": workout.completed,
            "created_at": workout.created_at.isoformat(),
        }

    def _dict_to_workout(self, data: Dict) -> Workout:
        """Convert dictionary to Workout"""
        return Workout(
            id=data["id"],
            activity_type=ActivityType(data["activity_type"]),
            date=date.fromisoformat(data["date"]),
            duration_minutes=data["duration_minutes"],
            intensity=IntensityLevel(data["intensity"]),
            calories_burned=data.get("calories_burned"),
            distance_km=data.get("distance_km"),
            heart_rate_avg=data.get("heart_rate_avg"),
            heart_rate_max=data.get("heart_rate_max"),
            notes=data.get("notes", ""),
            location=data.get("location", ""),
            completed=data.get("completed", True),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def _nutrition_to_dict(self, nutrition: NutritionLog) -> Dict:
        """Convert NutritionLog to dictionary"""
        return {
            "id": nutrition.id,
            "date": nutrition.date.isoformat(),
            "meal_type": nutrition.meal_type,
            "food_name": nutrition.food_name,
            "calories": nutrition.calories,
            "protein_g": nutrition.protein_g,
            "carbs_g": nutrition.carbs_g,
            "fat_g": nutrition.fat_g,
            "fiber_g": nutrition.fiber_g,
            "sugar_g": nutrition.sugar_g,
            "sodium_mg": nutrition.sodium_mg,
            "water_ml": nutrition.water_ml,
            "notes": nutrition.notes,
            "created_at": nutrition.created_at.isoformat(),
        }

    def _dict_to_nutrition(self, data: Dict) -> NutritionLog:
        """Convert dictionary to NutritionLog"""
        return NutritionLog(
            id=data["id"],
            date=date.fromisoformat(data["date"]),
            meal_type=data["meal_type"],
            food_name=data["food_name"],
            calories=data["calories"],
            protein_g=data.get("protein_g", 0),
            carbs_g=data.get("carbs_g", 0),
            fat_g=data.get("fat_g", 0),
            fiber_g=data.get("fiber_g", 0),
            sugar_g=data.get("sugar_g", 0),
            sodium_mg=data.get("sodium_mg", 0),
            water_ml=data.get("water_ml", 0),
            notes=data.get("notes", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def _metric_to_dict(self, metric: BodyMetric) -> Dict:
        """Convert BodyMetric to dictionary"""
        return {
            "id": metric.id,
            "date": metric.date.isoformat(),
            "weight_kg": metric.weight_kg,
            "body_fat_percent": metric.body_fat_percent,
            "muscle_mass_kg": metric.muscle_mass_kg,
            "bmi": metric.bmi,
            "waist_cm": metric.waist_cm,
            "hip_cm": metric.hip_cm,
            "chest_cm": metric.chest_cm,
            "arm_cm": metric.arm_cm,
            "thigh_cm": metric.thigh_cm,
            "notes": metric.notes,
            "created_at": metric.created_at.isoformat(),
        }

    def _dict_to_metric(self, data: Dict) -> BodyMetric:
        """Convert dictionary to BodyMetric"""
        return BodyMetric(
            id=data["id"],
            date=date.fromisoformat(data["date"]),
            weight_kg=data.get("weight_kg"),
            body_fat_percent=data.get("body_fat_percent"),
            muscle_mass_kg=data.get("muscle_mass_kg"),
            bmi=data.get("bmi"),
            waist_cm=data.get("waist_cm"),
            hip_cm=data.get("hip_cm"),
            chest_cm=data.get("chest_cm"),
            arm_cm=data.get("arm_cm"),
            thigh_cm=data.get("thigh_cm"),
            notes=data.get("notes", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def _goal_to_dict(self, goal: FitnessGoal) -> Dict:
        """Convert FitnessGoal to dictionary"""
        return {
            "id": goal.id,
            "type": goal.type.value,
            "target_value": goal.target_value,
            "current_value": goal.current_value,
            "unit": goal.unit,
            "start_date": goal.start_date.isoformat(),
            "target_date": goal.target_date.isoformat(),
            "progress": goal.progress,
            "completed": goal.completed,
            "created_at": goal.created_at.isoformat(),
        }

    def _dict_to_goal(self, data: Dict) -> FitnessGoal:
        """Convert dictionary to FitnessGoal"""
        return FitnessGoal(
            id=data["id"],
            type=GoalType(data["type"]),
            target_value=data["target_value"],
            current_value=data["current_value"],
            unit=data["unit"],
            start_date=date.fromisoformat(data["start_date"]),
            target_date=date.fromisoformat(data["target_date"]),
            progress=data.get("progress", 0),
            completed=data.get("completed", False),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def _update_stats(self):
        """Update statistics from data"""
        self.stats["total_workouts"] = len(self.workouts)
        self.stats["total_calories_burned"] = sum(
            w.calories_burned or 0 for w in self.workouts
        )
        self.stats["total_workout_minutes"] = sum(
            w.duration_minutes for w in self.workouts
        )
        self.stats["goals_completed"] = sum(1 for g in self.goals if g.completed)

    def _update_streak(self):
        """Update current workout streak"""
        if not self.workouts:
            self.stats["current_streak_days"] = 0
            return

        # Get unique workout dates
        workout_dates = sorted(set(w.date for w in self.workouts if w.completed))

        if not workout_dates:
            self.stats["current_streak_days"] = 0
            return

        # Calculate current streak from today backwards
        today = date.today()
        streak = 0
        check_date = today

        while check_date in workout_dates:
            streak += 1
            check_date -= timedelta(days=1)

        self.stats["current_streak_days"] = streak

        # Calculate longest streak
        longest = 0
        current = 1
        for i in range(1, len(workout_dates)):
            if (workout_dates[i] - workout_dates[i - 1]).days == 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
        longest = max(longest, current)

        self.stats["longest_streak_days"] = longest

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid

        return str(uuid.uuid4())[:8]

    def _calculate_calories_burned(
        self,
        activity_type: ActivityType,
        duration_minutes: int,
        intensity: IntensityLevel,
    ) -> int:
        """Calculate calories burned based on MET"""
        # Get MET value
        met = self.met_values.get(activity_type, 5.0)

        # Adjust MET based on intensity
        intensity_multipliers = {
            IntensityLevel.VERY_LOW: 0.7,
            IntensityLevel.LOW: 0.85,
            IntensityLevel.MODERATE: 1.0,
            IntensityLevel.HIGH: 1.15,
            IntensityLevel.VERY_HIGH: 1.3,
        }

        adjusted_met = met * intensity_multipliers.get(intensity, 1.0)

        # Calculate calories: MET * weight(kg) * duration(hours)
        weight_kg = self.user_profile.get("weight_kg", 70)
        duration_hours = duration_minutes / 60
        calories = adjusted_met * weight_kg * duration_hours

        return int(calories)

    def calculate_bmi(self, weight_kg: float, height_cm: float) -> float:
        """Calculate BMI"""
        height_m = height_cm / 100
        return weight_kg / (height_m**2)

    def calculate_bmr(self) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
        weight_kg = self.user_profile.get("weight_kg", 70)
        height_cm = self.user_profile.get("height_cm", 170)
        age = self.user_profile.get("age", 30)
        gender = self.user_profile.get("gender", "male")

        if gender == "male":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

        return round(bmr, 0)

    def calculate_tdee(self) -> float:
        """Calculate Total Daily Energy Expenditure"""
        bmr = self.calculate_bmr()
        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }

        activity_level = self.user_profile.get("activity_level", "moderate")
        multiplier = activity_multipliers.get(activity_level, 1.55)

        return round(bmr * multiplier, 0)

    # ============
    # Workout Management
    # ============

    async def log_workout(
        self,
        activity_type: Union[ActivityType, str],
        duration_minutes: int,
        date: Optional[date] = None,
        intensity: Union[IntensityLevel, str] = IntensityLevel.MODERATE,
        distance_km: Optional[float] = None,
        heart_rate_avg: Optional[int] = None,
        heart_rate_max: Optional[int] = None,
        notes: str = "",
        location: str = "",
    ) -> Dict[str, Any]:
        """
        Log a workout session

        Args:
            activity_type: Type of activity
            duration_minutes: Duration in minutes
            date: Date of workout (default today)
            intensity: Intensity level
            distance_km: Distance in kilometers
            heart_rate_avg: Average heart rate
            heart_rate_max: Maximum heart rate
            notes: Additional notes
            location: Workout location

        Returns:
            Dictionary with workout logging result
        """
        if date is None:
            date = datetime.now().date()

        if isinstance(activity_type, str):
            activity_type = ActivityType(activity_type.lower())

        if isinstance(intensity, str):
            intensity = IntensityLevel(intensity.lower())

        # Calculate calories burned
        calories_burned = self._calculate_calories_burned(
            activity_type, duration_minutes, intensity
        )

        workout = Workout(
            id=self._generate_id(),
            activity_type=activity_type,
            date=date,
            duration_minutes=duration_minutes,
            intensity=intensity,
            calories_burned=calories_burned,
            distance_km=distance_km,
            heart_rate_avg=heart_rate_avg,
            heart_rate_max=heart_rate_max,
            notes=notes,
            location=location,
        )

        self.workouts.append(workout)
        self._save_data()
        self._update_stats()
        self._update_streak()

        # Update goal progress
        await self._update_goal_progress()

        self.logger.info(
            f"Workout logged: {activity_type.value} for {duration_minutes} minutes"
        )

        return {
            "success": True,
            "workout_id": workout.id,
            "calories_burned": calories_burned,
            "message": "Workout logged successfully",
        }

    async def get_workouts(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        activity_type: Optional[ActivityType] = None,
    ) -> Dict[str, Any]:
        """Get workouts within date range"""
        filtered = self.workouts

        if start_date:
            filtered = [w for w in filtered if w.date >= start_date]
        if end_date:
            filtered = [w for w in filtered if w.date <= end_date]
        if activity_type:
            filtered = [w for w in filtered if w.activity_type == activity_type]

        # Sort by date descending
        filtered.sort(key=lambda x: x.date, reverse=True)

        return {
            "success": True,
            "total": len(filtered),
            "workouts": [
                {
                    "id": w.id,
                    "activity": w.activity_type.value,
                    "date": w.date.isoformat(),
                    "duration_minutes": w.duration_minutes,
                    "calories_burned": w.calories_burned,
                    "distance_km": w.distance_km,
                    "intensity": w.intensity.value,
                }
                for w in filtered
            ],
        }

    # ============
    # Nutrition Management
    # ============

    async def log_meal(
        self,
        meal_type: str,
        food_name: str,
        calories: int,
        date: Optional[date] = None,
        protein_g: float = 0,
        carbs_g: float = 0,
        fat_g: float = 0,
        fiber_g: float = 0,
        sugar_g: float = 0,
        water_ml: int = 0,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Log a meal entry

        Args:
            meal_type: Type of meal (breakfast, lunch, dinner, snack)
            food_name: Name of food
            calories: Calorie count
            date: Date of meal (default today)
            protein_g: Protein in grams
            carbs_g: Carbohydrates in grams
            fat_g: Fat in grams
            fiber_g: Fiber in grams
            sugar_g: Sugar in grams
            water_ml: Water consumed with meal
            notes: Additional notes

        Returns:
            Dictionary with meal logging result
        """
        if date is None:
            date = datetime.now().date()

        nutrition = NutritionLog(
            id=self._generate_id(),
            date=date,
            meal_type=meal_type,
            food_name=food_name,
            calories=calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
            fiber_g=fiber_g,
            sugar_g=sugar_g,
            water_ml=water_ml,
            notes=notes,
        )

        self.nutrition_logs.append(nutrition)
        self._save_data()

        self.logger.info(
            f"Meal logged: {meal_type} - {food_name} ({calories} calories)"
        )

        return {
            "success": True,
            "log_id": nutrition.id,
            "message": "Meal logged successfully",
        }

    async def log_water(
        self, amount_ml: int, date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Log water intake"""
        if date is None:
            date = datetime.now().date()

        # Create a water entry as a special meal
        return await self.log_meal(
            meal_type="water",
            food_name="Water",
            calories=0,
            date=date,
            water_ml=amount_ml,
            notes="Water intake",
        )

    async def get_nutrition_summary(
        self, target_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get nutrition summary for a date"""
        if target_date is None:
            target_date = datetime.now().date()

        day_logs = [n for n in self.nutrition_logs if n.date == target_date]

        total_calories = sum(n.calories for n in day_logs)
        total_protein = sum(n.protein_g for n in day_logs)
        total_carbs = sum(n.carbs_g for n in day_logs)
        total_fat = sum(n.fat_g for n in day_logs)
        total_fiber = sum(n.fiber_g for n in day_logs)
        total_sugar = sum(n.sugar_g for n in day_logs)
        total_water = sum(n.water_ml for n in day_logs)

        # Calculate macros percentage
        total_macros = total_protein + total_carbs + total_fat
        if total_macros > 0:
            protein_pct = (
                (total_protein * 4 / total_calories) * 100 if total_calories > 0 else 0
            )
            carbs_pct = (
                (total_carbs * 4 / total_calories) * 100 if total_calories > 0 else 0
            )
            fat_pct = (
                (total_fat * 9 / total_calories) * 100 if total_calories > 0 else 0
            )
        else:
            protein_pct = carbs_pct = fat_pct = 0

        return {
            "success": True,
            "date": target_date.isoformat(),
            "total_calories": total_calories,
            "total_protein_g": total_protein,
            "total_carbs_g": total_carbs,
            "total_fat_g": total_fat,
            "total_fiber_g": total_fiber,
            "total_sugar_g": total_sugar,
            "total_water_ml": total_water,
            "macros_percentage": {
                "protein": round(protein_pct, 1),
                "carbs": round(carbs_pct, 1),
                "fat": round(fat_pct, 1),
            },
            "meals": [
                {
                    "meal_type": n.meal_type,
                    "food_name": n.food_name,
                    "calories": n.calories,
                }
                for n in day_logs
            ],
        }

    # ============
    # Body Metrics
    # ============

    async def log_body_metrics(
        self,
        weight_kg: Optional[float] = None,
        body_fat_percent: Optional[float] = None,
        waist_cm: Optional[float] = None,
        hip_cm: Optional[float] = None,
        chest_cm: Optional[float] = None,
        arm_cm: Optional[float] = None,
        thigh_cm: Optional[float] = None,
        date: Optional[date] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Log body measurements

        Args:
            weight_kg: Weight in kilograms
            body_fat_percent: Body fat percentage
            waist_cm: Waist circumference in cm
            hip_cm: Hip circumference in cm
            chest_cm: Chest circumference in cm
            arm_cm: Arm circumference in cm
            thigh_cm: Thigh circumference in cm
            date: Date of measurement
            notes: Additional notes

        Returns:
            Dictionary with logging result
        """
        if date is None:
            date = datetime.now().date()

        # Calculate BMI if weight provided
        bmi = None
        if weight_kg:
            height_cm = self.user_profile.get("height_cm", 170)
            bmi = self.calculate_bmi(weight_kg, height_cm)

        metric = BodyMetric(
            id=self._generate_id(),
            date=date,
            weight_kg=weight_kg,
            body_fat_percent=body_fat_percent,
            bmi=bmi,
            waist_cm=waist_cm,
            hip_cm=hip_cm,
            chest_cm=chest_cm,
            arm_cm=arm_cm,
            thigh_cm=thigh_cm,
            notes=notes,
        )

        self.body_metrics.append(metric)
        self._save_data()

        # Update user profile weight
        if weight_kg:
            self.user_profile["weight_kg"] = weight_kg

        self.logger.info(f"Body metrics logged for {date}")

        return {
            "success": True,
            "metric_id": metric.id,
            "bmi": bmi,
            "message": "Body metrics logged successfully",
        }

    async def get_weight_progress(self, days: int = 30) -> Dict[str, Any]:
        """Get weight progress over time"""
        cutoff_date = datetime.now().date() - timedelta(days=days)
        metrics = [
            m for m in self.body_metrics if m.weight_kg and m.date >= cutoff_date
        ]
        metrics.sort(key=lambda x: x.date)

        if not metrics:
            return {"success": True, "data": [], "message": "No weight data available"}

        start_weight = metrics[0].weight_kg
        current_weight = metrics[-1].weight_kg
        change = current_weight - start_weight if start_weight else 0

        return {
            "success": True,
            "data": [
                {"date": m.date.isoformat(), "weight_kg": m.weight_kg, "bmi": m.bmi}
                for m in metrics
            ],
            "start_weight": start_weight,
            "current_weight": current_weight,
            "change": round(change, 1),
            "trend": "loss" if change < 0 else "gain" if change > 0 else "stable",
        }

    # ============
    # Goal Management
    # ============

    async def set_goal(
        self,
        goal_type: Union[GoalType, str],
        target_value: float,
        target_date: date,
        current_value: float = 0,
        unit: str = "",
    ) -> Dict[str, Any]:
        """
        Set a fitness goal

        Args:
            goal_type: Type of goal
            target_value: Target value to achieve
            target_date: Target completion date
            current_value: Current value
            unit: Unit of measurement

        Returns:
            Dictionary with goal setting result
        """
        if isinstance(goal_type, str):
            goal_type = GoalType(goal_type.lower())

        # Set default units
        unit_map = {
            GoalType.WEIGHT_LOSS: "kg",
            GoalType.MUSCLE_GAIN: "kg",
            GoalType.ENDURANCE: "minutes",
            GoalType.STRENGTH: "kg",
            GoalType.GENERAL_FITNESS: "workouts",
        }

        if not unit:
            unit = unit_map.get(goal_type, "units")

        goal = FitnessGoal(
            id=self._generate_id(),
            type=goal_type,
            target_value=target_value,
            current_value=current_value,
            unit=unit,
            start_date=datetime.now().date(),
            target_date=target_date,
        )

        self.goals.append(goal)
        self._save_data()

        self.logger.info(f"Goal set: {goal_type.value} - target {target_value} {unit}")

        return {
            "success": True,
            "goal_id": goal.id,
            "message": f"Goal set: {goal_type.value}",
        }

    async def _update_goal_progress(self):
        """Update progress for all active goals"""
        for goal in self.goals:
            if goal.completed:
                continue

            if goal.type == GoalType.WEIGHT_LOSS:
                # Get latest weight
                latest_weight = None
                for metric in sorted(
                    self.body_metrics, key=lambda x: x.date, reverse=True
                ):
                    if metric.weight_kg:
                        latest_weight = metric.weight_kg
                        break

                if latest_weight:
                    goal.current_value = latest_weight
                    if goal.target_value > goal.start_value:
                        # Weight loss goal
                        progress = (
                            (goal.start_value - latest_weight)
                            / (goal.start_value - goal.target_value)
                        ) * 100
                    else:
                        progress = 0
                    goal.progress = max(0, min(100, progress))

            elif goal.type == GoalType.GENERAL_FITNESS:
                # Count workouts
                workouts_count = len(
                    [w for w in self.workouts if w.date >= goal.start_date]
                )
                goal.current_value = workouts_count
                goal.progress = (workouts_count / goal.target_value) * 100

            # Check if completed
            if goal.current_value >= goal.target_value and goal.target_value > 0:
                goal.completed = True
                self.stats["goals_completed"] += 1

        self._save_data()

    async def get_goals(self, active_only: bool = True) -> Dict[str, Any]:
        """Get fitness goals"""
        goals = self.goals
        if active_only:
            goals = [g for g in goals if not g.completed]

        return {
            "success": True,
            "total": len(goals),
            "goals": [
                {
                    "id": g.id,
                    "type": g.type.value,
                    "target": g.target_value,
                    "current": g.current_value,
                    "unit": g.unit,
                    "progress": round(g.progress, 1),
                    "completed": g.completed,
                    "target_date": g.target_date.isoformat(),
                }
                for g in goals
            ],
        }

    # ============
    # Daily Summary
    # ============

    async def get_daily_summary(
        self, target_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get complete daily fitness summary"""
        if target_date is None:
            target_date = datetime.now().date()

        # Get workouts
        day_workouts = [
            w for w in self.workouts if w.date == target_date and w.completed
        ]

        # Get nutrition
        nutrition_summary = await self.get_nutrition_summary(target_date)

        # Calculate totals
        total_calories_burned = sum(w.calories_burned or 0 for w in day_workouts)
        total_workout_minutes = sum(w.duration_minutes for w in day_workouts)
        total_calories_consumed = nutrition_summary.get("total_calories", 0)

        # Calculate TDEE recommendation
        tdee = self.calculate_tdee()
        daily_target = tdee

        return {
            "success": True,
            "date": target_date.isoformat(),
            "workouts": {
                "count": len(day_workouts),
                "total_minutes": total_workout_minutes,
                "calories_burned": total_calories_burned,
                "activities": [w.activity_type.value for w in day_workouts],
            },
            "nutrition": {
                "calories_consumed": total_calories_consumed,
                "calories_remaining": daily_target - total_calories_consumed,
                "daily_target": daily_target,
                "water_ml": nutrition_summary.get("total_water_ml", 0),
            },
            "net_calories": total_calories_burned - total_calories_consumed,
            "streak": self.stats["current_streak_days"],
            "recommendation": self._get_daily_recommendation(
                total_calories_burned, total_calories_consumed, daily_target
            ),
        }

    def _get_daily_recommendation(self, burned: int, consumed: int, target: int) -> str:
        """Generate daily recommendation based on stats"""
        if consumed > target:
            return f"You're {consumed - target} calories over your target. Consider lighter meals tomorrow."
        elif consumed < target - 500:
            return f"You're {target - consumed} calories under your target. Make sure you're eating enough for energy."
        elif burned < 200:
            return "Consider adding some physical activity today to boost your fitness."
        else:
            return "Great job today! Keep up the good work!"

    # ============
    # Analytics and Reports
    # ============

    async def get_weekly_report(self) -> Dict[str, Any]:
        """Get weekly fitness report"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)

        week_workouts = [w for w in self.workouts if start_date <= w.date <= end_date]

        total_calories = sum(w.calories_burned or 0 for w in week_workouts)
        total_minutes = sum(w.duration_minutes for w in week_workouts)

        # Group by activity
        activity_breakdown = {}
        for workout in week_workouts:
            activity = workout.activity_type.value
            activity_breakdown[activity] = activity_breakdown.get(activity, 0) + 1

        return {
            "success": True,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_workouts": len(week_workouts),
            "total_minutes": total_minutes,
            "total_calories": total_calories,
            "average_per_day": round(total_minutes / 7, 1),
            "activity_breakdown": activity_breakdown,
            "streak": self.stats["current_streak_days"],
            "longest_streak": self.stats["longest_streak_days"],
        }

    async def get_workout_recommendation(self) -> Dict[str, Any]:
        """Get personalized workout recommendation"""
        # Analyze recent workouts
        recent_workouts = (
            self.workouts[-10:] if len(self.workouts) >= 10 else self.workouts
        )

        # Determine most common activities
        activity_counts = {}
        for w in recent_workouts:
            activity_counts[w.activity_type.value] = (
                activity_counts.get(w.activity_type.value, 0) + 1
            )

        # Suggest variety
        recommended = None
        if activity_counts:
            most_common = max(activity_counts, key=activity_counts.get)
            if activity_counts.get(most_common, 0) > 3:
                # Suggest different activity
                all_activities = [a.value for a in ActivityType]
                for activity in all_activities:
                    if activity_counts.get(activity, 0) == 0:
                        recommended = activity
                        break

        if not recommended:
            recommended = "mixed cardio and strength training"

        return {
            "success": True,
            "recommendation": f"Based on your recent activity, try {recommended} today for better balance.",
            "recent_activities": activity_counts,
        }

    async def get_bmr_tdee_info(self) -> Dict[str, Any]:
        """Get BMR and TDEE information"""
        bmr = self.calculate_bmr()
        tdee = self.calculate_tdee()

        return {
            "success": True,
            "bmr": bmr,
            "tdee": tdee,
            "weight_kg": self.user_profile.get("weight_kg"),
            "height_cm": self.user_profile.get("height_cm"),
            "age": self.user_profile.get("age"),
            "gender": self.user_profile.get("gender"),
            "activity_level": self.user_profile.get("activity_level"),
            "explanation": {
                "bmr": f"Your body burns approximately {bmr} calories per day at complete rest.",
                "tdee": f"With your activity level, you burn approximately {tdee} calories per day.",
                "weight_loss": f"To lose 0.5 kg per week, aim for {tdee - 500} calories per day.",
                "weight_gain": f"To gain 0.5 kg per week, aim for {tdee + 500} calories per day.",
            },
        }

    # ============
    # Utility Methods
    # ============

    async def update_user_profile(self, **updates) -> Dict[str, Any]:
        """Update user profile information"""
        for key, value in updates.items():
            if key in self.user_profile:
                self.user_profile[key] = value

        self._save_data()

        return {
            "success": True,
            "profile": self.user_profile,
            "message": "User profile updated",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "total_nutrition_logs": len(self.nutrition_logs),
            "total_body_metrics": len(self.body_metrics),
            "active_goals": len([g for g in self.goals if not g.completed]),
        }


# Integration wrapper for EDIATH
class FitnessAgentWrapper:
    """
    Wrapper class to integrate FitnessAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.fitness_agent = FitnessAgent(config)
        self.agent_type = "fitness"
        self.capabilities = [
            "log_workout",
            "log_meal",
            "log_body_metrics",
            "set_goal",
            "get_daily_summary",
            "get_weekly_report",
            "get_workout_recommendation",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a fitness request

        Request format:
        {
            'operation': 'log_workout|log_meal|log_metrics|set_goal|summary|report|recommend|stats',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "log_workout":
            return await self.fitness_agent.log_workout(
                activity_type=request.get("activity_type"),
                duration_minutes=request.get("duration_minutes"),
                intensity=request.get("intensity", "moderate"),
                distance_km=request.get("distance_km"),
                heart_rate_avg=request.get("heart_rate_avg"),
                heart_rate_max=request.get("heart_rate_max"),
                notes=request.get("notes", ""),
                location=request.get("location", ""),
            )

        elif operation == "log_meal":
            return await self.fitness_agent.log_meal(
                meal_type=request.get("meal_type"),
                food_name=request.get("food_name"),
                calories=request.get("calories"),
                protein_g=request.get("protein_g", 0),
                carbs_g=request.get("carbs_g", 0),
                fat_g=request.get("fat_g", 0),
                fiber_g=request.get("fiber_g", 0),
                sugar_g=request.get("sugar_g", 0),
                water_ml=request.get("water_ml", 0),
                notes=request.get("notes", ""),
            )

        elif operation == "log_water":
            return await self.fitness_agent.log_water(
                amount_ml=request.get("amount_ml"), date=request.get("date")
            )

        elif operation == "log_metrics":
            return await self.fitness_agent.log_body_metrics(
                weight_kg=request.get("weight_kg"),
                body_fat_percent=request.get("body_fat_percent"),
                waist_cm=request.get("waist_cm"),
                hip_cm=request.get("hip_cm"),
                chest_cm=request.get("chest_cm"),
                arm_cm=request.get("arm_cm"),
                thigh_cm=request.get("thigh_cm"),
                notes=request.get("notes", ""),
            )

        elif operation == "set_goal":
            return await self.fitness_agent.set_goal(
                goal_type=request.get("goal_type"),
                target_value=request.get("target_value"),
                target_date=datetime.fromisoformat(request.get("target_date")).date(),
                current_value=request.get("current_value", 0),
                unit=request.get("unit", ""),
            )

        elif operation == "summary":
            date_str = request.get("date")
            target_date = date.fromisoformat(date_str) if date_str else None
            return await self.fitness_agent.get_daily_summary(target_date)

        elif operation == "nutrition":
            date_str = request.get("date")
            target_date = date.fromisoformat(date_str) if date_str else None
            return await self.fitness_agent.get_nutrition_summary(target_date)

        elif operation == "weight_progress":
            return await self.fitness_agent.get_weight_progress(
                days=request.get("days", 30)
            )

        elif operation == "report":
            return await self.fitness_agent.get_weekly_report()

        elif operation == "recommend":
            return await self.fitness_agent.get_workout_recommendation()

        elif operation == "bmr_tdee":
            return await self.fitness_agent.get_bmr_tdee_info()

        elif operation == "goals":
            return await self.fitness_agent.get_goals(
                active_only=request.get("active_only", True)
            )

        elif operation == "workouts":
            start_date = request.get("start_date")
            end_date = request.get("end_date")
            return await self.fitness_agent.get_workouts(
                start_date=date.fromisoformat(start_date) if start_date else None,
                end_date=date.fromisoformat(end_date) if end_date else None,
            )

        elif operation == "update_profile":
            updates = {k: v for k, v in request.items() if k not in ["operation"]}
            return await self.fitness_agent.update_user_profile(**updates)

        elif operation == "stats":
            return self.fitness_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "FitnessAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.fitness_agent.get_stats(),
            "activity_types": [a.value for a in ActivityType],
            "intensity_levels": [i.value for i in IntensityLevel],
            "goal_types": [g.value for g in GoalType],
        }


# Example usage and testing
async def test_fitness_agent():
    """Test the fitness agent functionality"""

    # Initialize agent
    agent = FitnessAgent()

    print("=== Fitness Agent Test ===\n")

    # Test logging a workout
    print("1. Logging Workout...")
    result = await agent.log_workout(
        activity_type="running",
        duration_minutes=30,
        intensity="moderate",
        distance_km=5.0,
        heart_rate_avg=145,
    )
    print(f"   Workout logged: {result['success']}")
    print(f"   Calories burned: {result['calories_burned']}")

    # Test logging a meal
    print("\n2. Logging Meal...")
    result = await agent.log_meal(
        meal_type="lunch",
        food_name="Grilled Chicken Salad",
        calories=450,
        protein_g=35,
        carbs_g=20,
        fat_g=15,
    )
    print(f"   Meal logged: {result['success']}")

    # Test logging body metrics
    print("\n3. Logging Body Metrics...")
    result = await agent.log_body_metrics(
        weight_kg=72.5, body_fat_percent=18.5, waist_cm=82
    )
    print(f"   Metrics logged: {result['success']}")
    print(f"   BMI: {result['bmi']:.1f}")

    # Test getting daily summary
    print("\n4. Daily Summary...")
    result = await agent.get_daily_summary()
    if result["success"]:
        print(
            f"   Workouts: {result['workouts']['count']} ({result['workouts']['total_minutes']} min)"
        )
        print(
            f"   Calories: {result['nutrition']['calories_consumed']} consumed / {result['workouts']['calories_burned']} burned"
        )
        print(f"   Net: {result['net_calories']}")
        print(f"   Recommendation: {result['recommendation']}")

    # Test BMR/TDEE info
    print("\n5. BMR & TDEE Information...")
    result = await agent.get_bmr_tdee_info()
    if result["success"]:
        print(f"   BMR: {result['bmr']} calories/day")
        print(f"   TDEE: {result['tdee']} calories/day")
        print(f"   Weight loss target: {result['explanation']['weight_loss']}")

    # Test setting a goal
    print("\n6. Setting Fitness Goal...")
    result = await agent.set_goal(
        goal_type="weight_loss",
        target_value=70.0,
        target_date=(datetime.now() + timedelta(days=60)).date(),
        current_value=72.5,
        unit="kg",
    )
    print(f"   Goal set: {result['success']}")

    # Test weekly report
    print("\n7. Weekly Report...")
    result = await agent.get_weekly_report()
    if result["success"]:
        print(f"   Total workouts: {result['total_workouts']}")
        print(f"   Total minutes: {result['total_minutes']}")
        print(f"   Current streak: {result['streak']} days")

    # Test workout recommendation
    print("\n8. Workout Recommendation...")
    result = await agent.get_workout_recommendation()
    if result["success"]:
        print(f"   {result['recommendation']}")

    # Get statistics
    print("\n9. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total workouts: {stats['total_workouts']}")
    print(f"   Total calories burned: {stats['total_calories_burned']}")
    print(f"   Current streak: {stats['current_streak_days']} days")
    print(f"   Goals completed: {stats['goals_completed']}")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_fitness_agent())
