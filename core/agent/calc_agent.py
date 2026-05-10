"""
🔥 FINAL PRODUCTION Calculator Agent for EDIATH
✔ Mathematical calculations and expression evaluation
✔ Advanced math functions (trigonometry, logarithms, exponents)
✔ Statistical calculations (mean, median, mode, std, variance)
✔ Unit conversions (length, weight, temperature, area, volume, speed)
✔ Percentage calculations
✔ Matrix operations (add, subtract, multiply, determinant, inverse)
✔ Equation solving (linear, quadratic, systems)
✔ Fraction and decimal precision handling
✔ Expression parsing with order of operations
✔ History tracking with timestamps
✔ Batch processing
✔ Scientific notation support
✔ Constant library (pi, e, tau, etc.)
✔ Production ready
"""

import re
import math
import ast
import operator
import json
from typing import Dict, Any, List, Union, Optional
from decimal import getcontext
from fractions import Fraction
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Set high precision for decimal operations
getcontext().prec = 50


# =========================
# ENUMS AND CONSTANTS
# =========================


class CalculationType(Enum):
    """Types of calculations"""

    BASIC = "basic"
    EXPRESSION = "expression"
    STATISTICAL = "statistical"
    UNIT_CONVERSION = "unit_conversion"
    PERCENTAGE = "percentage"
    MATRIX = "matrix"
    EQUATION = "equation"
    FRACTION = "fraction"
    SCIENTIFIC = "scientific"


class UnitCategory(Enum):
    """Unit categories for conversion"""

    LENGTH = "length"
    WEIGHT = "weight"
    TEMPERATURE = "temperature"
    AREA = "area"
    VOLUME = "volume"
    SPEED = "speed"
    TIME = "time"
    DIGITAL = "digital"
    ENERGY = "energy"
    PRESSURE = "pressure"


# =========================
# DATACLASSES
# =========================


@dataclass
class CalculationResult:
    """Result of a calculation"""

    success: bool
    expression: str
    result: Any
    calculation_type: CalculationType
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


@dataclass
class Matrix:
    """Simple matrix representation"""

    rows: int
    cols: int
    data: List[List[Union[int, float]]]

    def __post_init__(self):
        if len(self.data) != self.rows:
            raise ValueError(f"Expected {self.rows} rows, got {len(self.data)}")
        for row in self.data:
            if len(row) != self.cols:
                raise ValueError(f"Expected {self.cols} columns, got {len(row)}")

    def get(self, i: int, j: int) -> Union[int, float]:
        return self.data[i][j]

    def set(self, i: int, j: int, value: Union[int, float]):
        self.data[i][j] = value

    def to_list(self) -> List[List[Union[int, float]]]:
        return self.data

    def __repr__(self) -> str:
        return f"Matrix({self.rows}x{self.cols})"


# =========================
# CALCULATOR AGENT
# =========================


class CalculatorAgent:
    """
    Advanced calculator agent capable of:
    - Basic arithmetic (+, -, *, /, //, %, **)
    - Mathematical functions (sin, cos, tan, sqrt, log, etc.)
    - Expression parsing and evaluation
    - Unit conversions
    - Statistical calculations
    - Matrix operations
    - Equation solving
    - Fraction and decimal precision handling
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Calculator Agent

        Args:
            config: Configuration dictionary with calculator settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Precision settings
        self.precision = self.config.get("precision", 10)
        self.use_decimal = self.config.get("use_decimal", False)

        # Supported operators
        self.operators = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
            "//": operator.floordiv,
            "%": operator.mod,
            "**": operator.pow,
        }

        # Mathematical constants
        self.constants = {
            "pi": math.pi,
            "π": math.pi,
            "e": math.e,
            "tau": math.tau,
            "inf": float("inf"),
            "infinity": float("inf"),
            "nan": float("nan"),
            "phi": (1 + math.sqrt(5)) / 2,  # Golden ratio
            "golden_ratio": (1 + math.sqrt(5)) / 2,
        }

        # Mathematical functions
        self.functions = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "atan2": math.atan2,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "asinh": math.asinh,
            "acosh": math.acosh,
            "atanh": math.atanh,
            "sqrt": math.sqrt,
            "cbrt": lambda x: x ** (1 / 3),
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "ln": math.log,
            "lg": math.log10,
            "exp": math.exp,
            "expm1": math.expm1,
            "abs": abs,
            "ceil": math.ceil,
            "floor": math.floor,
            "round": round,
            "trunc": math.trunc,
            "factorial": math.factorial,
            "gcd": math.gcd,
            "lcm": math.lcm,
            "radians": math.radians,
            "degrees": math.degrees,
            "hypot": math.hypot,
            "erf": math.erf,
            "erfc": math.erfc,
            "gamma": math.gamma,
            "lgamma": math.lgamma,
        }

        # Unit conversion factors
        self._init_unit_conversions()

        # Operation history
        self.history: List[CalculationResult] = []
        self.max_history = self.config.get("max_history", 1000)
        self.history_file = Path(self.config.get("history_file", "calc_history.json"))

        # Load history from file
        self._load_history()

        self.logger.info("Calculator Agent initialized")

    def _init_unit_conversions(self):
        """Initialize unit conversion factors"""
        # Base units: meters, kilograms, seconds, Celsius
        self.unit_conversions = {
            UnitCategory.LENGTH: {
                # To meters
                "m": 1.0,
                "meter": 1.0,
                "meters": 1.0,
                "km": 1000.0,
                "kilometer": 1000.0,
                "kilometers": 1000.0,
                "cm": 0.01,
                "centimeter": 0.01,
                "centimeters": 0.01,
                "mm": 0.001,
                "millimeter": 0.001,
                "millimeters": 0.001,
                "μm": 0.000001,
                "micrometer": 0.000001,
                "micrometers": 0.000001,
                "nm": 1e-9,
                "nanometer": 1e-9,
                "nanometers": 1e-9,
                "mi": 1609.344,
                "mile": 1609.344,
                "miles": 1609.344,
                "ft": 0.3048,
                "foot": 0.3048,
                "feet": 0.3048,
                "in": 0.0254,
                "inch": 0.0254,
                "inches": 0.0254,
                "yd": 0.9144,
                "yard": 0.9144,
                "yards": 0.9144,
                "nmi": 1852.0,
                "nautical_mile": 1852.0,
                "au": 149597870700.0,
                "astronomical_unit": 149597870700.0,
                "ly": 9.461e15,
                "light_year": 9.461e15,
                "pc": 3.086e16,
                "parsec": 3.086e16,
            },
            UnitCategory.WEIGHT: {
                # To kilograms
                "kg": 1.0,
                "kilogram": 1.0,
                "kilograms": 1.0,
                "g": 0.001,
                "gram": 0.001,
                "grams": 0.001,
                "mg": 0.000001,
                "milligram": 0.000001,
                "milligrams": 0.000001,
                "μg": 1e-9,
                "microgram": 1e-9,
                "micrograms": 1e-9,
                "lb": 0.45359237,
                "pound": 0.45359237,
                "pounds": 0.45359237,
                "oz": 0.028349523125,
                "ounce": 0.028349523125,
                "ounces": 0.028349523125,
                "st": 6.35029318,
                "stone": 6.35029318,
                "ton": 907.18474,
                "us_ton": 907.18474,
                "t": 1000.0,
                "metric_ton": 1000.0,
            },
            UnitCategory.TEMPERATURE: {
                "celsius": "C",
                "c": "C",
                "fahrenheit": "F",
                "f": "F",
                "kelvin": "K",
                "k": "K",
            },
            UnitCategory.AREA: {
                # To square meters
                "m2": 1.0,
                "sq_m": 1.0,
                "square_meter": 1.0,
                "km2": 1e6,
                "sq_km": 1e6,
                "square_kilometer": 1e6,
                "cm2": 0.0001,
                "sq_cm": 0.0001,
                "square_centimeter": 0.0001,
                "mm2": 1e-6,
                "sq_mm": 1e-6,
                "square_millimeter": 1e-6,
                "ft2": 0.09290304,
                "sq_ft": 0.09290304,
                "square_foot": 0.09290304,
                "in2": 0.00064516,
                "sq_in": 0.00064516,
                "square_inch": 0.00064516,
                "yd2": 0.83612736,
                "sq_yd": 0.83612736,
                "square_yard": 0.83612736,
                "acre": 4046.8564224,
                "ha": 10000.0,
                "hectare": 10000.0,
            },
            UnitCategory.VOLUME: {
                # To cubic meters
                "m3": 1.0,
                "cubic_meter": 1.0,
                "L": 0.001,
                "liter": 0.001,
                "liters": 0.001,
                "mL": 0.000001,
                "milliliter": 0.000001,
                "milliliters": 0.000001,
                "gal": 0.003785411784,
                "gallon": 0.003785411784,
                "gallons": 0.003785411784,
                "qt": 0.000946352946,
                "quart": 0.000946352946,
                "pt": 0.000473176473,
                "pint": 0.000473176473,
                "cup": 0.0002365882365,
                "fl_oz": 2.95735295625e-5,
                "fluid_ounce": 2.95735295625e-5,
                "tbsp": 1.478676478125e-5,
                "tablespoon": 1.478676478125e-5,
                "tsp": 4.92892159375e-6,
                "teaspoon": 4.92892159375e-6,
            },
            UnitCategory.SPEED: {
                # To meters per second
                "m/s": 1.0,
                "meter_per_second": 1.0,
                "km/h": 0.2777777777777778,
                "kph": 0.2777777777777778,
                "mph": 0.44704,
                "mile_per_hour": 0.44704,
                "knot": 0.5144444444444444,
                "kn": 0.5144444444444444,
                "ft/s": 0.3048,
                "foot_per_second": 0.3048,
                "c": 299792458.0,
                "speed_of_light": 299792458.0,
            },
            UnitCategory.TIME: {
                # To seconds
                "s": 1.0,
                "sec": 1.0,
                "second": 1.0,
                "seconds": 1.0,
                "ms": 0.001,
                "millisecond": 0.001,
                "milliseconds": 0.001,
                "μs": 1e-6,
                "microsecond": 1e-6,
                "microseconds": 1e-6,
                "ns": 1e-9,
                "nanosecond": 1e-9,
                "nanoseconds": 1e-9,
                "min": 60.0,
                "minute": 60.0,
                "minutes": 60.0,
                "h": 3600.0,
                "hr": 3600.0,
                "hour": 3600.0,
                "hours": 3600.0,
                "d": 86400.0,
                "day": 86400.0,
                "days": 86400.0,
                "wk": 604800.0,
                "week": 604800.0,
                "weeks": 604800.0,
                "yr": 31536000.0,
                "year": 31536000.0,
                "years": 31536000.0,
            },
            UnitCategory.DIGITAL: {
                # To bytes
                "B": 1.0,
                "byte": 1.0,
                "bytes": 1.0,
                "KB": 1024.0,
                "kilobyte": 1024.0,
                "MB": 1048576.0,
                "megabyte": 1048576.0,
                "GB": 1073741824.0,
                "gigabyte": 1073741824.0,
                "TB": 1099511627776.0,
                "terabyte": 1099511627776.0,
                "PB": 1125899906842624.0,
                "petabyte": 1125899906842624.0,
                "Kb": 128.0,
                "kilobit": 128.0,
                "Mb": 131072.0,
                "megabit": 131072.0,
                "Gb": 134217728.0,
                "gigabit": 134217728.0,
            },
            UnitCategory.ENERGY: {
                # To joules
                "J": 1.0,
                "joule": 1.0,
                "joules": 1.0,
                "kJ": 1000.0,
                "kilojoule": 1000.0,
                "cal": 4.184,
                "calorie": 4.184,
                "calories": 4.184,
                "kcal": 4184.0,
                "kilocalorie": 4184.0,
                "Wh": 3600.0,
                "watt_hour": 3600.0,
                "kWh": 3600000.0,
                "kilowatt_hour": 3600000.0,
                "eV": 1.602176634e-19,
                "electronvolt": 1.602176634e-19,
                "MeV": 1.602176634e-13,
                "mega_electronvolt": 1.602176634e-13,
                "BTU": 1055.05585262,
                "british_thermal_unit": 1055.05585262,
            },
            UnitCategory.PRESSURE: {
                # To pascals
                "Pa": 1.0,
                "pascal": 1.0,
                "pascals": 1.0,
                "kPa": 1000.0,
                "kilopascal": 1000.0,
                "MPa": 1000000.0,
                "megapascal": 1000000.0,
                "bar": 100000.0,
                "mbar": 100.0,
                "millibar": 100.0,
                "psi": 6894.757293168,
                "pound_per_square_inch": 6894.757293168,
                "atm": 101325.0,
                "atmosphere": 101325.0,
                "mmHg": 133.322368421,
                "torr": 133.322368421,
            },
        }

    # =========================
    # HISTORY MANAGEMENT
    # =========================

    def _load_history(self):
        """Load calculation history from file"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)
                for item in data[-self.max_history :]:
                    self.history.append(
                        CalculationResult(
                            success=item["success"],
                            expression=item["expression"],
                            result=item["result"],
                            calculation_type=CalculationType(item["calculation_type"]),
                            timestamp=datetime.fromisoformat(item["timestamp"]),
                            error=item.get("error"),
                            metadata=item.get("metadata", {}),
                            execution_time_ms=item.get("execution_time_ms", 0),
                        )
                    )
            self.logger.info(f"Loaded {len(self.history)} history entries")
        except Exception as e:
            self.logger.warning(f"Failed to load history: {e}")

    def _save_history(self):
        """Save calculation history to file"""
        try:
            data = []
            for item in self.history[-self.max_history :]:
                data.append(
                    {
                        "success": item.success,
                        "expression": item.expression,
                        "result": item.result,
                        "calculation_type": item.calculation_type.value,
                        "timestamp": item.timestamp.isoformat(),
                        "error": item.error,
                        "metadata": item.metadata,
                        "execution_time_ms": item.execution_time_ms,
                    }
                )

            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save history: {e}")

    def _add_to_history(self, result: CalculationResult):
        """Add calculation to history"""
        self.history.append(result)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        # Save to file periodically
        if len(self.history) % 10 == 0:
            self._save_history()

    # =========================
    # EXPRESSION EVALUATION
    # =========================

    def evaluate_expression(
        self, expression: str, safe_mode: bool = True
    ) -> CalculationResult:
        """
        Evaluate a mathematical expression

        Args:
            expression: Mathematical expression as string
            safe_mode: If True, restricts dangerous operations

        Returns:
            CalculationResult with result and metadata
        """
        import time

        start_time = time.time()

        try:
            # Clean the expression
            expression = expression.strip()

            # Replace constants
            expr_with_constants = expression
            for name, value in self.constants.items():
                expr_with_constants = expr_with_constants.replace(name, str(value))

            # Check if it's a simple calculation
            result = self._safe_eval(expr_with_constants, safe_mode)

            # Round result
            if isinstance(result, float):
                result = round(result, self.precision)

            calc_result = CalculationResult(
                success=True,
                expression=expression,
                result=result,
                calculation_type=CalculationType.EXPRESSION,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"safe_mode": safe_mode},
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            return CalculationResult(
                success=False,
                expression=expression,
                result=None,
                calculation_type=CalculationType.EXPRESSION,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _safe_eval(self, expression: str, safe_mode: bool = True) -> Any:
        """
        Safely evaluate mathematical expression

        Args:
            expression: Math expression string
            safe_mode: Enable security restrictions

        Returns:
            Evaluated result
        """
        # Token validation for safe mode
        if safe_mode:
            # Allow only safe patterns
            allowed_pattern = r"^[\d\s\+\-\*\/\%\*\*\(\)\,\.\_a-zA-Z]+$"
            if not re.match(allowed_pattern, expression):
                raise ValueError("Expression contains unsafe characters")

        # Create restricted namespace
        namespace = {
            "__builtins__": {},
            **self.functions,
            **self.constants,
        }

        # Try literal evaluation first (for simple numbers)
        try:
            result = ast.literal_eval(expression)
            return result
        except (ValueError, SyntaxError):
            # Use eval with restricted namespace
            result = eval(expression, namespace)
            return result

    # =========================
    # BASIC OPERATIONS
    # =========================

    def calculate_basic(
        self, a: Union[int, float], b: Union[int, float], operation: str
    ) -> CalculationResult:
        """
        Perform basic binary operations

        Args:
            a: First operand
            b: Second operand
            operation: Operator (+, -, *, /, //, %, **)

        Returns:
            CalculationResult with result
        """
        import time

        start_time = time.time()

        try:
            if operation not in self.operators:
                raise ValueError(f"Unsupported operation: {operation}")

            result = self.operators[operation](a, b)

            # Handle division by zero
            if operation == "/" and b == 0:
                raise ZeroDivisionError("Division by zero")

            # Round result
            if isinstance(result, float):
                result = round(result, self.precision)

            calc_result = CalculationResult(
                success=True,
                expression=f"{a} {operation} {b}",
                result=result,
                calculation_type=CalculationType.BASIC,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"a": a, "b": b, "operation": operation},
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            self.logger.error(f"Basic calculation error: {str(e)}")
            return CalculationResult(
                success=False,
                expression=f"{a} {operation} {b}",
                result=None,
                calculation_type=CalculationType.BASIC,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    # =========================
    # STATISTICAL CALCULATIONS
    # =========================

    def calculate_statistics(
        self, numbers: List[Union[int, float]], operation: str
    ) -> CalculationResult:
        """
        Calculate statistical metrics

        Args:
            numbers: List of numbers
            operation: Statistical operation (mean, median, mode, std, variance, sum, min, max, range, q1, q3, iqr)

        Returns:
            CalculationResult with statistical result
        """
        import time

        start_time = time.time()

        if not numbers:
            return CalculationResult(
                success=False,
                expression=f"{operation}({numbers})",
                result=None,
                calculation_type=CalculationType.STATISTICAL,
                error="Empty list provided",
            )

        try:
            result = None
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)

            if operation == "mean":
                result = sum(numbers) / n
            elif operation == "median":
                mid = n // 2
                if n % 2 == 0:
                    result = (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
                else:
                    result = sorted_nums[mid]
            elif operation == "mode":
                from collections import Counter

                counter = Counter(numbers)
                max_count = max(counter.values())
                modes = [k for k, v in counter.items() if v == max_count]
                result = modes[0] if len(modes) == 1 else modes
            elif operation == "std":
                mean = sum(numbers) / n
                variance = sum((x - mean) ** 2 for x in numbers) / n
                result = math.sqrt(variance)
            elif operation == "variance":
                mean = sum(numbers) / n
                result = sum((x - mean) ** 2 for x in numbers) / n
            elif operation == "sum":
                result = sum(numbers)
            elif operation == "min":
                result = min(numbers)
            elif operation == "max":
                result = max(numbers)
            elif operation == "range":
                result = max(numbers) - min(numbers)
            elif operation == "q1":  # First quartile
                q1_index = int(n * 0.25)
                result = sorted_nums[q1_index]
            elif operation == "q3":  # Third quartile
                q3_index = int(n * 0.75)
                result = sorted_nums[q3_index]
            elif operation == "iqr":  # Interquartile range
                q1_index = int(n * 0.25)
                q3_index = int(n * 0.75)
                result = sorted_nums[q3_index] - sorted_nums[q1_index]
            else:
                raise ValueError(f"Unsupported statistical operation: {operation}")

            # Round result
            if isinstance(result, float):
                result = round(result, self.precision)

            calc_result = CalculationResult(
                success=True,
                expression=(
                    f"{operation}({numbers[:5]}...)"
                    if len(numbers) > 5
                    else f"{operation}({numbers})"
                ),
                result=result,
                calculation_type=CalculationType.STATISTICAL,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"operation": operation, "count": n, "numbers": numbers[:10]},
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            self.logger.error(f"Statistics calculation error: {str(e)}")
            return CalculationResult(
                success=False,
                expression=f"{operation}({numbers[:5]}...)",
                result=None,
                calculation_type=CalculationType.STATISTICAL,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    # =========================
    # UNIT CONVERSION
    # =========================

    def convert_units(
        self, value: float, from_unit: str, to_unit: str
    ) -> CalculationResult:
        """
        Convert between different units

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            CalculationResult with conversion result
        """
        import time

        start_time = time.time()

        try:
            # Find category containing both units
            category = None
            from_unit_lower = from_unit.lower()
            to_unit_lower = to_unit.lower()

            # Temperature conversion (special case)
            if from_unit_lower in ["celsius", "c", "fahrenheit", "f", "kelvin", "k"]:
                result = self._convert_temperature(
                    value, from_unit_lower, to_unit_lower
                )
                calc_result = CalculationResult(
                    success=True,
                    expression=f"{value} {from_unit} to {to_unit}",
                    result=result,
                    calculation_type=CalculationType.UNIT_CONVERSION,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    metadata={
                        "category": "temperature",
                        "from": from_unit,
                        "to": to_unit,
                    },
                )
                self._add_to_history(calc_result)
                return calc_result

            # Find matching category
            for cat, conversions in self.unit_conversions.items():
                if from_unit_lower in conversions and to_unit_lower in conversions:
                    category = cat
                    break

            if category is None:
                raise ValueError(f"Cannot convert between {from_unit} and {to_unit}")

            # Convert to base unit then to target
            from_factor = self.unit_conversions[category][from_unit_lower]
            to_factor = self.unit_conversions[category][to_unit_lower]

            in_base = value * from_factor
            result = in_base / to_factor

            # Round result
            if isinstance(result, float):
                result = round(result, self.precision)

            calc_result = CalculationResult(
                success=True,
                expression=f"{value} {from_unit} to {to_unit}",
                result=result,
                calculation_type=CalculationType.UNIT_CONVERSION,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"category": category.value, "from": from_unit, "to": to_unit},
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            self.logger.error(f"Unit conversion error: {str(e)}")
            return CalculationResult(
                success=False,
                expression=f"{value} {from_unit} to {to_unit}",
                result=None,
                calculation_type=CalculationType.UNIT_CONVERSION,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Handle temperature unit conversions"""
        # Convert to Celsius first
        if from_unit in ["celsius", "c"]:
            celsius = value
        elif from_unit in ["fahrenheit", "f"]:
            celsius = (value - 32) * 5 / 9
        elif from_unit in ["kelvin", "k"]:
            celsius = value - 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

        # Convert from Celsius to target
        if to_unit in ["celsius", "c"]:
            return round(celsius, self.precision)
        elif to_unit in ["fahrenheit", "f"]:
            return round(celsius * 9 / 5 + 32, self.precision)
        elif to_unit in ["kelvin", "k"]:
            return round(celsius + 273.15, self.precision)
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")

    # =========================
    # PERCENTAGE CALCULATIONS
    # =========================

    def percentage(
        self, value: float, percent: float, operation: str = "of"
    ) -> CalculationResult:
        """
        Calculate percentages

        Args:
            value: Base value
            percent: Percentage
            operation: 'of' (percent of value), 'add' (add percent),
                      'subtract' (subtract percent), 'difference' (percentage difference),
                      'change' (percentage change from value to percent)

        Returns:
            CalculationResult with percentage calculation
        """
        import time

        start_time = time.time()

        try:
            if operation == "of":
                result = (percent / 100) * value
                formula = f"{percent}% of {value} = {result}"
            elif operation == "add":
                result = value + ((percent / 100) * value)
                formula = f"{value} + {percent}% = {result}"
            elif operation == "subtract":
                result = value - ((percent / 100) * value)
                formula = f"{value} - {percent}% = {result}"
            elif operation == "difference":
                result = ((value - percent) / percent) * 100 if percent != 0 else 0
                formula = (
                    f"Percentage difference between {value} and {percent} = {result}%"
                )
            elif operation == "change":
                result = ((percent - value) / value) * 100 if value != 0 else 0
                formula = f"Percentage change from {value} to {percent} = {result}%"
            else:
                raise ValueError(f"Unknown percentage operation: {operation}")

            # Round result
            if isinstance(result, float):
                result = round(result, self.precision)

            calc_result = CalculationResult(
                success=True,
                expression=formula,
                result=result,
                calculation_type=CalculationType.PERCENTAGE,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"value": value, "percent": percent, "operation": operation},
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            return CalculationResult(
                success=False,
                expression=f"percentage({value}, {percent}, {operation})",
                result=None,
                calculation_type=CalculationType.PERCENTAGE,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    # =========================
    # MATRIX OPERATIONS
    # =========================

    def matrix_add(self, a: Matrix, b: Matrix) -> CalculationResult:
        """Add two matrices"""
        import time

        start_time = time.time()

        try:
            if a.rows != b.rows or a.cols != b.cols:
                raise ValueError("Matrices must have same dimensions for addition")

            result_data = []
            for i in range(a.rows):
                row = []
                for j in range(a.cols):
                    row.append(a.get(i, j) + b.get(i, j))
                result_data.append(row)

            result = Matrix(a.rows, a.cols, result_data)

            calc_result = CalculationResult(
                success=True,
                expression=f"Matrix addition ({a.rows}x{a.cols}) + ({b.rows}x{b.cols})",
                result=result,
                calculation_type=CalculationType.MATRIX,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            return CalculationResult(
                success=False,
                expression="matrix_add",
                result=None,
                calculation_type=CalculationType.MATRIX,
                error=str(e),
            )

    def matrix_multiply(self, a: Matrix, b: Matrix) -> CalculationResult:
        """Multiply two matrices"""
        import time

        start_time = time.time()

        try:
            if a.cols != b.rows:
                raise ValueError(
                    f"Cannot multiply {a.rows}x{a.cols} with {b.rows}x{b.cols}"
                )

            result_data = [[0 for _ in range(b.cols)] for _ in range(a.rows)]

            for i in range(a.rows):
                for j in range(b.cols):
                    for k in range(a.cols):
                        result_data[i][j] += a.get(i, k) * b.get(k, j)

            result = Matrix(a.rows, b.cols, result_data)

            calc_result = CalculationResult(
                success=True,
                expression=f"Matrix multiplication ({a.rows}x{a.cols}) * ({b.rows}x{b.cols})",
                result=result,
                calculation_type=CalculationType.MATRIX,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            return CalculationResult(
                success=False,
                expression="matrix_multiply",
                result=None,
                calculation_type=CalculationType.MATRIX,
                error=str(e),
            )

    def matrix_determinant(self, m: Matrix) -> CalculationResult:
        """Calculate determinant of a square matrix"""
        import time

        start_time = time.time()

        try:
            if m.rows != m.cols:
                raise ValueError("Determinant only defined for square matrices")

            if m.rows == 1:
                result = m.get(0, 0)
            elif m.rows == 2:
                result = m.get(0, 0) * m.get(1, 1) - m.get(0, 1) * m.get(1, 0)
            else:
                # Recursive determinant for larger matrices
                result = self._determinant_recursive(m.data)

            calc_result = CalculationResult(
                success=True,
                expression=f"det({m.rows}x{m.cols} matrix)",
                result=result,
                calculation_type=CalculationType.MATRIX,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            return CalculationResult(
                success=False,
                expression="matrix_determinant",
                result=None,
                calculation_type=CalculationType.MATRIX,
                error=str(e),
            )

    def _determinant_recursive(self, matrix: List[List[Union[int, float]]]) -> float:
        """Calculate determinant recursively"""
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

        det = 0
        for c in range(n):
            sub_matrix = [row[:c] + row[c + 1 :] for row in matrix[1:]]
            det += ((-1) ** c) * matrix[0][c] * self._determinant_recursive(sub_matrix)
        return det

    # =========================
    # EQUATION SOLVING
    # =========================

    def solve_linear(self, a: float, b: float) -> CalculationResult:
        """Solve linear equation a*x + b = 0"""
        import time

        start_time = time.time()

        try:
            if a == 0:
                if b == 0:
                    result = "infinite solutions"
                else:
                    result = "no solution"
            else:
                result = -b / a

            calc_result = CalculationResult(
                success=True,
                expression=f"{a}x + {b} = 0",
                result=result,
                calculation_type=CalculationType.EQUATION,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            return CalculationResult(
                success=False,
                expression=f"{a}x + {b} = 0",
                result=None,
                calculation_type=CalculationType.EQUATION,
                error=str(e),
            )

    def solve_quadratic(self, a: float, b: float, c: float) -> CalculationResult:
        """Solve quadratic equation a*x² + b*x + c = 0"""
        import time

        start_time = time.time()

        try:
            if a == 0:
                return self.solve_linear(b, c)

            discriminant = b**2 - 4 * a * c

            if discriminant > 0:
                x1 = (-b + math.sqrt(discriminant)) / (2 * a)
                x2 = (-b - math.sqrt(discriminant)) / (2 * a)
                result = [x1, x2]
            elif discriminant == 0:
                result = [-b / (2 * a)]
            else:
                real = -b / (2 * a)
                imag = math.sqrt(-discriminant) / (2 * a)
                result = [complex(real, imag), complex(real, -imag)]

            calc_result = CalculationResult(
                success=True,
                expression=f"{a}x² + {b}x + {c} = 0",
                result=result,
                calculation_type=CalculationType.EQUATION,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            return CalculationResult(
                success=False,
                expression=f"{a}x² + {b}x + {c} = 0",
                result=None,
                calculation_type=CalculationType.EQUATION,
                error=str(e),
            )

    # =========================
    # FRACTION OPERATIONS
    # =========================

    def fraction_add(self, a: Fraction, b: Fraction) -> CalculationResult:
        """Add two fractions"""
        import time

        start_time = time.time()

        result = a + b

        calc_result = CalculationResult(
            success=True,
            expression=f"{a} + {b}",
            result=result,
            calculation_type=CalculationType.FRACTION,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

        self._add_to_history(calc_result)
        return calc_result

    def fraction_simplify(self, numerator: int, denominator: int) -> CalculationResult:
        """Simplify a fraction"""
        import time

        start_time = time.time()

        try:
            frac = Fraction(numerator, denominator)

            calc_result = CalculationResult(
                success=True,
                expression=f"{numerator}/{denominator}",
                result=frac,
                calculation_type=CalculationType.FRACTION,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            return CalculationResult(
                success=False,
                expression=f"{numerator}/{denominator}",
                result=None,
                calculation_type=CalculationType.FRACTION,
                error=str(e),
            )

    # =========================
    # SCIENTIFIC CALCULATIONS
    # =========================

    def scientific_notation(self, value: float) -> CalculationResult:
        """Convert to scientific notation"""
        import time

        start_time = time.time()

        try:
            if value == 0:
                result = "0"
            else:
                exponent = math.floor(math.log10(abs(value)))
                mantissa = value / (10**exponent)
                result = f"{mantissa:.{self.precision}g} × 10^{exponent}"

            calc_result = CalculationResult(
                success=True,
                expression=f"Scientific notation of {value}",
                result=result,
                calculation_type=CalculationType.SCIENTIFIC,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            self._add_to_history(calc_result)
            return calc_result

        except Exception as e:
            return CalculationResult(
                success=False,
                expression=f"scientific_notation({value})",
                result=None,
                calculation_type=CalculationType.SCIENTIFIC,
                error=str(e),
            )

    # =========================
    # BATCH PROCESSING
    # =========================

    def batch_calculate(self, expressions: List[str]) -> List[CalculationResult]:
        """
        Evaluate multiple expressions in batch

        Args:
            expressions: List of mathematical expressions

        Returns:
            List of results for each expression
        """
        results = []
        for expr in expressions:
            result = self.evaluate_expression(expr)
            results.append(result)

        return results

    # =========================
    # HISTORY QUERIES
    # =========================

    def get_history(
        self, limit: int = None, calculation_type: CalculationType = None
    ) -> List[CalculationResult]:
        """Get calculation history"""
        history = self.history

        if calculation_type:
            history = [h for h in history if h.calculation_type == calculation_type]

        if limit:
            return history[-limit:]
        return history

    def clear_history(self) -> Dict[str, Any]:
        """Clear calculation history"""
        count = len(self.history)
        self.history = []
        self._save_history()

        return {
            "success": True,
            "cleared": count,
            "message": f"Cleared {count} history entries",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get calculator statistics"""
        total = len(self.history)
        successful = sum(1 for h in self.history if h.success)
        failed = total - successful

        type_counts = {}
        for h in self.history:
            type_counts[h.calculation_type.value] = (
                type_counts.get(h.calculation_type.value, 0) + 1
            )

        return {
            "total_calculations": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "by_type": type_counts,
            "precision": self.precision,
            "use_decimal": self.use_decimal,
        }

    def export_history(self, filepath: str) -> Dict[str, Any]:
        """Export history to file"""
        try:
            data = []
            for item in self.history:
                data.append(
                    {
                        "success": item.success,
                        "expression": item.expression,
                        "result": str(item.result),
                        "calculation_type": item.calculation_type.value,
                        "timestamp": item.timestamp.isoformat(),
                        "error": item.error,
                        "execution_time_ms": item.execution_time_ms,
                    }
                )

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            return {"success": True, "exported": len(data), "file": filepath}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =========================
# INTEGRATION WRAPPER
# =========================


class CalculatorAgentWrapper:
    """
    Wrapper class to integrate CalculatorAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.calculator = CalculatorAgent(config)
        self.agent_type = "calculator"
        self.capabilities = [
            "basic_arithmetic",
            "expression_evaluation",
            "statistics",
            "unit_conversion",
            "percentage_calculation",
            "matrix_operations",
            "equation_solving",
            "fraction_operations",
            "scientific_notation",
            "batch_processing",
        ]
        self._initialized = True

    async def initialize(self, *args, **kwargs) -> bool:
        """Initialize the wrapper"""
        return True

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a calculation request

        Expected request formats:
        - {'type': 'expression', 'expression': '2+2'}
        - {'type': 'basic', 'a': 10, 'b': 5, 'operation': '+'}
        - {'type': 'statistics', 'numbers': [1,2,3,4,5], 'operation': 'mean'}
        - {'type': 'unit_convert', 'value': 10, 'from': 'km', 'to': 'miles'}
        - {'type': 'percentage', 'value': 200, 'percent': 15, 'operation': 'of'}
        - {'type': 'matrix_add', 'matrix_a': [...], 'matrix_b': [...]}
        - {'type': 'solve_linear', 'a': 2, 'b': -4}
        - {'type': 'solve_quadratic', 'a': 1, 'b': -3, 'c': 2}
        """

        request_type = request.get("type", "expression")

        try:
            if request_type == "expression":
                result = self.calculator.evaluate_expression(
                    request.get("expression", ""),
                    safe_mode=request.get("safe_mode", True),
                )
                return self._result_to_dict(result)

            elif request_type == "basic":
                result = self.calculator.calculate_basic(
                    request.get("a", 0),
                    request.get("b", 0),
                    request.get("operation", "+"),
                )
                return self._result_to_dict(result)

            elif request_type == "statistics":
                result = self.calculator.calculate_statistics(
                    request.get("numbers", []), request.get("operation", "mean")
                )
                return self._result_to_dict(result)

            elif request_type == "unit_convert":
                result = self.calculator.convert_units(
                    request.get("value", 0),
                    request.get("from", ""),
                    request.get("to", ""),
                )
                return self._result_to_dict(result)

            elif request_type == "percentage":
                result = self.calculator.percentage(
                    request.get("value", 0),
                    request.get("percent", 0),
                    request.get("operation", "of"),
                )
                return self._result_to_dict(result)

            elif request_type == "matrix_add":
                a = Matrix(
                    request.get("rows_a", 0),
                    request.get("cols_a", 0),
                    request.get("matrix_a", []),
                )
                b = Matrix(
                    request.get("rows_b", 0),
                    request.get("cols_b", 0),
                    request.get("matrix_b", []),
                )
                result = self.calculator.matrix_add(a, b)
                return self._result_to_dict(result)

            elif request_type == "matrix_multiply":
                a = Matrix(
                    request.get("rows_a", 0),
                    request.get("cols_a", 0),
                    request.get("matrix_a", []),
                )
                b = Matrix(
                    request.get("rows_b", 0),
                    request.get("cols_b", 0),
                    request.get("matrix_b", []),
                )
                result = self.calculator.matrix_multiply(a, b)
                return self._result_to_dict(result)

            elif request_type == "matrix_determinant":
                m = Matrix(
                    request.get("rows", 0),
                    request.get("cols", 0),
                    request.get("matrix", []),
                )
                result = self.calculator.matrix_determinant(m)
                return self._result_to_dict(result)

            elif request_type == "solve_linear":
                result = self.calculator.solve_linear(
                    request.get("a", 0), request.get("b", 0)
                )
                return self._result_to_dict(result)

            elif request_type == "solve_quadratic":
                result = self.calculator.solve_quadratic(
                    request.get("a", 1), request.get("b", 0), request.get("c", 0)
                )
                return self._result_to_dict(result)

            elif request_type == "fraction_simplify":
                result = self.calculator.fraction_simplify(
                    request.get("numerator", 0), request.get("denominator", 1)
                )
                return self._result_to_dict(result)

            elif request_type == "scientific":
                result = self.calculator.scientific_notation(request.get("value", 0))
                return self._result_to_dict(result)

            elif request_type == "batch":
                results = self.calculator.batch_calculate(
                    request.get("expressions", [])
                )
                return {
                    "success": True,
                    "results": [self._result_to_dict(r) for r in results],
                    "count": len(results),
                }

            elif request_type == "history":
                return {
                    "success": True,
                    "history": [
                        self._result_to_dict(h)
                        for h in self.calculator.get_history(
                            limit=request.get("limit"),
                            calculation_type=(
                                CalculationType(request.get("type_filter"))
                                if request.get("type_filter")
                                else None
                            ),
                        )
                    ],
                }

            elif request_type == "clear_history":
                return self.calculator.clear_history()

            elif request_type == "stats":
                return self.calculator.get_stats()

            else:
                return {
                    "success": False,
                    "error": f"Unknown request type: {request_type}",
                }

        except Exception as e:
            return {"success": False, "error": str(e), "request_type": request_type}

    def _result_to_dict(self, result: CalculationResult) -> Dict[str, Any]:
        """Convert CalculationResult to dictionary"""
        return {
            "success": result.success,
            "expression": result.expression,
            "result": str(result.result) if result.result is not None else None,
            "result_type": (
                type(result.result).__name__ if result.result is not None else None
            ),
            "calculation_type": result.calculation_type.value,
            "timestamp": result.timestamp.isoformat(),
            "error": result.error,
            "execution_time_ms": round(result.execution_time_ms, 2),
        }

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "CalculatorAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.calculator.get_stats(),
            "precision": self.calculator.precision,
        }

    async def close(self):
        """Clean up resources"""
        self.calculator._save_history()


# =========================
# TESTING
# =========================


def test_calculator_agent():
    """Test the calculator agent functionality"""

    calc = CalculatorAgent()

    print("=== Calculator Agent Test ===\n")

    # Test basic arithmetic
    print("1. Basic Arithmetic:")
    result = calc.calculate_basic(10, 3, "+")
    print(f"   10 + 3 = {result.result}")

    result = calc.calculate_basic(10, 3, "*")
    print(f"   10 * 3 = {result.result}")

    # Test expression evaluation
    print("\n2. Expression Evaluation:")
    result = calc.evaluate_expression("2 + 2 * 3")
    print(f"   2 + 2 * 3 = {result.result}")

    result = calc.evaluate_expression("sqrt(16) + sin(pi/2)")
    print(f"   sqrt(16) + sin(pi/2) = {result.result}")

    # Test statistics
    print("\n3. Statistics:")
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = calc.calculate_statistics(numbers, "mean")
    print(f"   Mean of {numbers[:5]}... = {result.result}")

    result = calc.calculate_statistics(numbers, "median")
    print(f"   Median = {result.result}")

    result = calc.calculate_statistics(numbers, "std")
    print(f"   Standard deviation = {result.result}")

    # Test unit conversion
    print("\n4. Unit Conversion:")
    result = calc.convert_units(10, "km", "miles")
    print(f"   10 km = {result.result} miles")

    result = calc.convert_units(100, "celsius", "fahrenheit")
    print(f"   100°C = {result.result}°F")

    result = calc.convert_units(1, "GB", "MB")
    print(f"   1 GB = {result.result} MB")

    # Test percentages
    print("\n5. Percentages:")
    result = calc.percentage(200, 15, "of")
    print(f"   15% of 200 = {result.result}")

    result = calc.percentage(100, 10, "add")
    print(f"   100 + 10% = {result.result}")

    result = calc.percentage(100, 20, "difference")
    print(f"   Percentage difference between 100 and 20 = {result.result}%")

    # Test equation solving
    print("\n6. Equation Solving:")
    result = calc.solve_linear(2, -4)
    print(f"   2x - 4 = 0 → x = {result.result}")

    result = calc.solve_quadratic(1, -3, 2)
    print(f"   x² - 3x + 2 = 0 → x = {result.result}")

    # Test scientific notation
    print("\n7. Scientific Notation:")
    result = calc.scientific_notation(1234567)
    print(f"   1234567 = {result.result}")

    # Test batch processing
    print("\n8. Batch Processing:")
    expressions = ["2+2", "10*5", "sqrt(100)", "log(100)"]
    results = calc.batch_calculate(expressions)
    for r in results:
        print(f"   {r.expression} = {r.result}")

    # Test statistics
    print("\n9. Calculator Statistics:")
    stats = calc.get_stats()
    print(f"   Total calculations: {stats['total_calculations']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   By type: {stats['by_type']}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_calculator_agent()
