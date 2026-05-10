"""
Math Reasoning Agent for EDIATH
Advanced mathematical reasoning: symbolic math, equation solving, calculus, linear algebra, statistics, proofs
"""

import asyncio
import math
import cmath
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import logging

# Symbolic mathematics
try:
    import sympy as sp
    from sympy import symbols, simplify, solve, diff, integrate, limit, series
    from sympy.parsing.sympy_parser import parse_expr

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Numerical computing
try:
    import numpy as np

    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

try:
    import scipy
    from scipy import optimize, integrate as scipy_integrate, stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Statistical analysis
try:
    import pandas as pd

    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

# Plotting (optional)
try:
    import matplotlib.pyplot as plt

    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


class MathDomain(Enum):
    """Mathematical domains"""

    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    STATISTICS = "statistics"
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    NUMBER_THEORY = "number_theory"
    DISCRETE_MATH = "discrete_math"
    DIFFERENTIAL_EQUATIONS = "differential_equations"


class ProblemType(Enum):
    """Problem types"""

    SIMPLIFY = "simplify"
    SOLVE = "solve"
    DERIVE = "derive"
    INTEGRATE = "integrate"
    LIMIT = "limit"
    MATRIX = "matrix"
    STATS = "statistics"
    PROOF = "proof"
    OPTIMIZATION = "optimization"
    SEQUENCE = "sequence"


@dataclass
class MathSolution:
    """Mathematical solution container"""

    problem: str
    solution: Any
    steps: List[str]
    domain: MathDomain
    problem_type: ProblemType
    confidence: float
    execution_time: float
    alternative_solutions: List[Any] = field(default_factory=list)
    latex: Optional[str] = None
    numerical_value: Optional[float] = None


@dataclass
class StatisticalResult:
    """Statistical analysis result"""

    data: List[float]
    mean: float
    median: float
    mode: List[float]
    variance: float
    std_dev: float
    min_val: float
    max_val: float
    quartiles: Dict[str, float]
    outliers: List[float]


class MathReasoningAgent:
    """
    Advanced mathematical reasoning agent capable of:
    - Symbolic algebra (simplification, expansion, factoring)
    - Equation solving (linear, polynomial, systems, differential)
    - Calculus (derivatives, integrals, limits, series)
    - Linear algebra (matrices, vectors, eigenvalues)
    - Statistics (descriptive, probability distributions)
    - Geometry calculations
    - Number theory (primes, GCD, LCM, modular arithmetic)
    - Discrete mathematics (combinatorics, graph theory)
    - Optimization (linear programming, curve fitting)
    - Step-by-step solution generation
    - LaTeX output formatting
    - Numerical approximations
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Math Reasoning Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Symbolic engine
        self.use_sympy = self.config.get("use_sympy", True) and SYMPY_AVAILABLE
        self.use_numpy = self.config.get("use_numpy", True) and NP_AVAILABLE
        self.use_scipy = self.config.get("use_scipy", True) and SCIPY_AVAILABLE

        # Precision settings
        self.precision = self.config.get("precision", 10)
        self.use_rational = self.config.get("use_rational", True)

        # Output settings
        self.show_steps = self.config.get("show_steps", True)
        self.output_latex = self.config.get("output_latex", True)

        # Cache
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        self.solution_cache: Dict[str, tuple] = {}

        # Constants
        self.constants = {
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "inf": float("inf"),
            "phi": (1 + math.sqrt(5)) / 2,  # Golden ratio
        }

        # Statistics
        self.stats = {
            "total_problems": 0,
            "solved_problems": 0,
            "failed_problems": 0,
            "average_time": 0.0,
            "by_domain": {},
        }

        # History
        self.solution_history: List[MathSolution] = []
        self.max_history = self.config.get("max_history", 100)

        self.logger.info(
            f"Math Reasoning Agent initialized (SymPy: {self.use_sympy}, NumPy: {self.use_numpy})"
        )

    def _get_cache_key(self, problem: str, domain: str, problem_type: str) -> str:
        """Generate cache key for problem"""
        import hashlib

        key_data = f"{problem}:{domain}:{problem_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[MathSolution]:
        """Get solution from cache"""
        if not self.cache_enabled:
            return None

        if cache_key in self.solution_cache:
            timestamp, solution = self.solution_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return solution
            else:
                del self.solution_cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, solution: MathSolution):
        """Add solution to cache"""
        if self.cache_enabled:
            if len(self.solution_cache) > 1000:
                items = sorted(self.solution_cache.items(), key=lambda x: x[1][0])
                for key, _ in items[:100]:
                    del self.solution_cache[key]

            self.solution_cache[cache_key] = (datetime.now(), solution)

    def _add_to_history(self, solution: MathSolution):
        """Add solution to history"""
        self.solution_history.append(solution)
        if len(self.solution_history) > self.max_history:
            self.solution_history = self.solution_history[-self.max_history :]

    async def solve(
        self,
        problem: str,
        domain: Optional[MathDomain] = None,
        problem_type: Optional[ProblemType] = None,
        variables: Optional[Dict] = None,
        show_steps: bool = True,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Solve a mathematical problem

        Args:
            problem: Mathematical problem as string
            domain: Mathematical domain (auto-detected if None)
            problem_type: Type of problem (auto-detected if None)
            variables: Variable values for substitution
            show_steps: Include step-by-step solution
            use_cache: Use cached results

        Returns:
            Dictionary with solution
        """
        start_time = datetime.now()

        # Auto-detect domain and type if not provided
        if domain is None or problem_type is None:
            detected_domain, detected_type = self._detect_problem_type(problem)
            domain = domain or detected_domain
            problem_type = problem_type or detected_type

        # Check cache
        cache_key = self._get_cache_key(problem, domain.value, problem_type.value)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                return self._format_solution(cached, start_time)

        try:
            # Route to appropriate solver
            if domain == MathDomain.ALGEBRA:
                solution = await self._solve_algebra(
                    problem, problem_type, variables, show_steps
                )
            elif domain == MathDomain.CALCULUS:
                solution = await self._solve_calculus(
                    problem, problem_type, variables, show_steps
                )
            elif domain == MathDomain.LINEAR_ALGEBRA:
                solution = await self._solve_linear_algebra(
                    problem, problem_type, variables, show_steps
                )
            elif domain == MathDomain.STATISTICS:
                solution = await self._solve_statistics(
                    problem, problem_type, variables, show_steps
                )
            elif domain == MathDomain.NUMBER_THEORY:
                solution = await self._solve_number_theory(
                    problem, problem_type, variables, show_steps
                )
            elif domain == MathDomain.GEOMETRY:
                solution = await self._solve_geometry(
                    problem, problem_type, variables, show_steps
                )
            elif domain == MathDomain.TRIGONOMETRY:
                solution = await self._solve_trigonometry(
                    problem, problem_type, variables, show_steps
                )
            elif domain == MathDomain.DISCRETE_MATH:
                solution = await self._solve_discrete(
                    problem, problem_type, variables, show_steps
                )
            else:
                solution = await self._solve_generic(
                    problem, problem_type, variables, show_steps
                )

            execution_time = (datetime.now() - start_time).total_seconds()
            solution.execution_time = execution_time

            # Update statistics
            self.stats["total_problems"] += 1
            self.stats["solved_problems"] += 1
            self.stats["average_time"] = (
                self.stats["average_time"] * (self.stats["total_problems"] - 1)
                + execution_time
            ) / self.stats["total_problems"]

            domain_name = domain.value
            if domain_name not in self.stats["by_domain"]:
                self.stats["by_domain"][domain_name] = 0
            self.stats["by_domain"][domain_name] += 1

            # Add to history
            self._add_to_history(solution)

            # Cache solution
            self._add_to_cache(cache_key, solution)

            return self._format_solution(solution, start_time)

        except Exception as e:
            self.logger.error(f"Math solving error: {str(e)}")
            self.stats["failed_problems"] += 1

            return {
                "success": False,
                "error": str(e),
                "problem": problem,
                "domain": domain.value if domain else "unknown",
                "problem_type": problem_type.value if problem_type else "unknown",
            }

    def _detect_problem_type(self, problem: str) -> Tuple[MathDomain, ProblemType]:
        """Detect problem domain and type from string"""
        problem_lower = problem.lower()

        # Domain detection
        if any(
            word in problem_lower
            for word in [
                "derivative",
                "differentiate",
                "d/dx",
                "integral",
                "integrate",
                "limit",
                "lim",
                "series",
            ]
        ):
            domain = MathDomain.CALCULUS
        elif any(
            word in problem_lower
            for word in [
                "matrix",
                "vector",
                "determinant",
                "eigenvalue",
                "linear algebra",
            ]
        ):
            domain = MathDomain.LINEAR_ALGEBRA
        elif any(
            word in problem_lower
            for word in [
                "mean",
                "median",
                "mode",
                "variance",
                "standard deviation",
                "probability",
                "distribution",
            ]
        ):
            domain = MathDomain.STATISTICS
        elif any(
            word in problem_lower
            for word in ["prime", "gcd", "lcm", "modulo", "congruence", "number theory"]
        ):
            domain = MathDomain.NUMBER_THEORY
        elif any(
            word in problem_lower
            for word in [
                "triangle",
                "circle",
                "area",
                "perimeter",
                "volume",
                "geometry",
            ]
        ):
            domain = MathDomain.GEOMETRY
        elif any(
            word in problem_lower
            for word in ["sin", "cos", "tan", "trigonometric", "angle"]
        ):
            domain = MathDomain.TRIGONOMETRY
        elif any(
            word in problem_lower
            for word in ["combination", "permutation", "graph", "set", "discrete"]
        ):
            domain = MathDomain.DISCRETE_MATH
        elif any(
            word in problem_lower
            for word in ["solve", "equation", "=", "simplify", "expand", "factor"]
        ):
            domain = MathDomain.ALGEBRA
        else:
            domain = MathDomain.ARITHMETIC

        # Problem type detection
        if "simplify" in problem_lower:
            problem_type = ProblemType.SIMPLIFY
        elif "solve" in problem_lower or "=" in problem:
            problem_type = ProblemType.SOLVE
        elif "derivative" in problem_lower or "differentiate" in problem_lower:
            problem_type = ProblemType.DERIVE
        elif "integral" in problem_lower or "integrate" in problem_lower:
            problem_type = ProblemType.INTEGRATE
        elif "limit" in problem_lower:
            problem_type = ProblemType.LIMIT
        elif any(word in problem_lower for word in ["matrix", "determinant", "eigen"]):
            problem_type = ProblemType.MATRIX
        elif any(word in problem_lower for word in ["mean", "median", "variance"]):
            problem_type = ProblemType.STATS
        else:
            problem_type = ProblemType.SIMPLIFY

        return domain, problem_type

    # ============
    # Algebra Solver
    # ============

    async def _solve_algebra(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Solve algebra problems"""
        steps = []

        if not self.use_sympy:
            return await self._solve_numerical(
                problem, problem_type, variables, show_steps
            )

        try:
            # Parse expression
            expr = parse_expr(problem)
            steps.append(f"Parsed expression: {sp.latex(expr)}")

            if problem_type == ProblemType.SIMPLIFY:
                result = sp.simplify(expr)
                steps.append(f"Simplified expression: {sp.latex(result)}")

                # Alternative simplifications
                alternatives = []
                try:
                    alt1 = sp.expand(expr)
                    if alt1 != result:
                        alternatives.append(alt1)
                except:
                    pass

                solution = MathSolution(
                    problem=problem,
                    solution=result,
                    steps=steps if show_steps else [],
                    domain=MathDomain.ALGEBRA,
                    problem_type=problem_type,
                    confidence=0.95,
                    alternative_solutions=alternatives,
                    latex=sp.latex(result),
                )

            elif problem_type == ProblemType.SOLVE:
                # Find variables in expression
                symbols_list = list(expr.free_symbols)
                if not symbols_list:
                    symbols_list = [sp.Symbol("x")]

                result = sp.solve(expr, symbols_list[0])
                steps.append(f"Solving for {symbols_list[0]}: {sp.latex(result)}")

                solution = MathSolution(
                    problem=problem,
                    solution=result,
                    steps=steps if show_steps else [],
                    domain=MathDomain.ALGEBRA,
                    problem_type=problem_type,
                    confidence=0.9,
                    latex=sp.latex(result),
                )

            else:
                # Generic simplification
                result = sp.simplify(expr)
                solution = MathSolution(
                    problem=problem,
                    solution=result,
                    steps=steps if show_steps else [],
                    domain=MathDomain.ALGEBRA,
                    problem_type=problem_type,
                    confidence=0.85,
                    latex=sp.latex(result),
                )

            # Try to get numerical value
            try:
                numerical = float(result.evalf())
                solution.numerical_value = numerical
            except:
                pass

            return solution

        except Exception as e:
            raise Exception(f"Algebra solving failed: {str(e)}")

    # ============
    # Calculus Solver
    # ============

    async def _solve_calculus(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Solve calculus problems"""
        steps = []

        if not self.use_sympy:
            return await self._solve_numerical(
                problem, problem_type, variables, show_steps
            )

        try:
            # Extract function and variable
            x = sp.Symbol("x")
            expr = parse_expr(problem)

            if problem_type == ProblemType.DERIVE:
                # Find derivative variable
                var = x
                if "d/d" in problem:
                    # Extract variable from d/dx notation
                    match = re.search(r"d/d([a-zA-Z])", problem)
                    if match:
                        var = sp.Symbol(match.group(1))

                result = sp.diff(expr, var)
                steps.append(f"Derivative with respect to {var}: {sp.latex(result)}")

                # Second derivative
                try:
                    second_deriv = sp.diff(expr, var, 2)
                    steps.append(f"Second derivative: {sp.latex(second_deriv)}")
                except:
                    pass

                solution = MathSolution(
                    problem=problem,
                    solution=result,
                    steps=steps if show_steps else [],
                    domain=MathDomain.CALCULUS,
                    problem_type=problem_type,
                    confidence=0.95,
                    latex=sp.latex(result),
                )

            elif problem_type == ProblemType.INTEGRATE:
                result = sp.integrate(expr, x)
                steps.append(f"Indefinite integral: {sp.latex(result)} + C")

                # Check for definite integral
                if "from" in problem.lower() or "to" in problem.lower():
                    bounds = re.findall(r"from\s+(\d+)\s+to\s+(\d+)", problem.lower())
                    if bounds:
                        a, b = float(bounds[0][0]), float(bounds[0][1])
                        definite = sp.integrate(expr, (x, a, b))
                        steps.append(
                            f"Definite integral from {a} to {b}: {sp.latex(definite)}"
                        )
                        result = definite

                solution = MathSolution(
                    problem=problem,
                    solution=result,
                    steps=steps if show_steps else [],
                    domain=MathDomain.CALCULUS,
                    problem_type=problem_type,
                    confidence=0.95,
                    latex=sp.latex(result),
                )

            elif problem_type == ProblemType.LIMIT:
                # Parse limit
                match = re.search(r"lim_{([a-zA-Z])->([^}]+)}", problem)
                if match:
                    var = sp.Symbol(match.group(1))
                    point = parse_expr(match.group(2))
                    result = sp.limit(expr, var, point)
                    steps.append(f"Limit as {var} → {point}: {sp.latex(result)}")

                    # Check from both sides
                    left_limit = sp.limit(expr, var, point, dir="-")
                    right_limit = sp.limit(expr, var, point, dir="+")
                    if left_limit != right_limit:
                        steps.append(f"Left-hand limit: {sp.latex(left_limit)}")
                        steps.append(f"Right-hand limit: {sp.latex(right_limit)}")
                else:
                    result = sp.limit(expr, x, 0)
                    steps.append(f"Limit as x → 0: {sp.latex(result)}")

                solution = MathSolution(
                    problem=problem,
                    solution=result,
                    steps=steps if show_steps else [],
                    domain=MathDomain.CALCULUS,
                    problem_type=problem_type,
                    confidence=0.9,
                    latex=sp.latex(result),
                )

            else:
                result = sp.simplify(expr)
                solution = MathSolution(
                    problem=problem,
                    solution=result,
                    steps=steps if show_steps else [],
                    domain=MathDomain.CALCULUS,
                    problem_type=problem_type,
                    confidence=0.85,
                    latex=sp.latex(result),
                )

            # Get numerical approximation
            try:
                numerical = float(result.evalf())
                solution.numerical_value = numerical
            except:
                pass

            return solution

        except Exception as e:
            raise Exception(f"Calculus solving failed: {str(e)}")

    # ============
    # Linear Algebra Solver
    # ============

    async def _solve_linear_algebra(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Solve linear algebra problems"""
        steps = []

        if not self.use_sympy:
            return await self._solve_numerical(
                problem, problem_type, variables, show_steps
            )

        try:
            # Parse matrix notation
            matrix_pattern = r"\[\[(.*?)\]\]"
            matrices = re.findall(matrix_pattern, problem)

            if "determinant" in problem.lower() or "det" in problem.lower():
                # Find matrix
                if matrices:
                    matrix_data = [
                        [float(x) for x in row.split(",")]
                        for row in matrices[0].split("],[")
                    ]
                    matrix = sp.Matrix(matrix_data)
                    result = matrix.det()
                    steps.append(f"Matrix determinant: {sp.latex(result)}")

                    solution = MathSolution(
                        problem=problem,
                        solution=result,
                        steps=steps if show_steps else [],
                        domain=MathDomain.LINEAR_ALGEBRA,
                        problem_type=problem_type,
                        confidence=0.95,
                        latex=sp.latex(result),
                    )
                else:
                    raise Exception("Matrix not found")

            elif "eigenvalue" in problem.lower() or "eigenvector" in problem.lower():
                if matrices:
                    matrix_data = [
                        [float(x) for x in row.split(",")]
                        for row in matrices[0].split("],[")
                    ]
                    matrix = sp.Matrix(matrix_data)
                    eigenvalues = matrix.eigenvals()
                    eigenvectors = matrix.eigenvects()

                    steps.append(f"Eigenvalues: {eigenvalues}")
                    steps.append(
                        f"Eigenvectors: {[(val, [v for v in vec]) for val, mult, vec in eigenvectors]}"
                    )

                    solution = MathSolution(
                        problem=problem,
                        solution={
                            "eigenvalues": eigenvalues,
                            "eigenvectors": eigenvectors,
                        },
                        steps=steps if show_steps else [],
                        domain=MathDomain.LINEAR_ALGEBRA,
                        problem_type=problem_type,
                        confidence=0.9,
                    )
                else:
                    raise Exception("Matrix not found")

            elif "inverse" in problem.lower():
                if matrices:
                    matrix_data = [
                        [float(x) for x in row.split(",")]
                        for row in matrices[0].split("],[")
                    ]
                    matrix = sp.Matrix(matrix_data)
                    result = matrix.inv()
                    steps.append(f"Inverse matrix: {sp.latex(result)}")

                    solution = MathSolution(
                        problem=problem,
                        solution=result,
                        steps=steps if show_steps else [],
                        domain=MathDomain.LINEAR_ALGEBRA,
                        problem_type=problem_type,
                        confidence=0.95,
                        latex=sp.latex(result),
                    )
                else:
                    raise Exception("Matrix not found")

            else:
                # Generic matrix operation
                if matrices:
                    matrix_data = [
                        [float(x) for x in row.split(",")]
                        for row in matrices[0].split("],[")
                    ]
                    matrix = sp.Matrix(matrix_data)
                    result = matrix

                    solution = MathSolution(
                        problem=problem,
                        solution=result,
                        steps=steps if show_steps else [],
                        domain=MathDomain.LINEAR_ALGEBRA,
                        problem_type=problem_type,
                        confidence=0.9,
                        latex=sp.latex(result),
                    )
                else:
                    raise Exception("Matrix operation not recognized")

            return solution

        except Exception as e:
            raise Exception(f"Linear algebra solving failed: {str(e)}")

    # ============
    # Statistics Solver
    # ============

    async def _solve_statistics(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Solve statistics problems"""
        steps = []

        # Extract data from problem
        data_pattern = r"\[([\d\.,\s]+)\]"
        match = re.search(data_pattern, problem)

        if not match:
            # Try comma-separated values
            numbers = re.findall(r"\d+\.?\d*", problem)
            data = [float(n) for n in numbers] if numbers else []
        else:
            data = [float(x.strip()) for x in match.group(1).split(",")]

        if not data:
            raise Exception("No data found in problem")

        if NP_AVAILABLE:
            arr = np.array(data)

            if "mean" in problem.lower():
                result = np.mean(arr)
                steps.append(
                    f"Mean = sum(data) / n = {sum(data)} / {len(data)} = {result}"
                )

            elif "median" in problem.lower():
                result = np.median(arr)
                steps.append(f"Median = middle value when sorted = {result}")

            elif "mode" in problem.lower():
                from collections import Counter

                counter = Counter(data)
                max_count = max(counter.values())
                result = [k for k, v in counter.items() if v == max_count]
                steps.append(f"Mode = most frequent value(s): {result}")

            elif "variance" in problem.lower():
                result = (
                    np.var(arr, ddof=1) if "sample" in problem.lower() else np.var(arr)
                )
                steps.append(f"Variance = average of squared deviations = {result}")

            elif "std" in problem.lower() or "standard deviation" in problem.lower():
                result = (
                    np.std(arr, ddof=1) if "sample" in problem.lower() else np.std(arr)
                )
                steps.append(f"Standard deviation = sqrt(variance) = {result}")

            elif "quartile" in problem.lower():
                q1 = np.percentile(arr, 25)
                q2 = np.percentile(arr, 50)
                q3 = np.percentile(arr, 75)
                result = {"Q1": q1, "Q2": q2, "Q3": q3}
                steps.append(f"Q1 (25th percentile): {q1}")
                steps.append(f"Q2 (50th percentile/median): {q2}")
                steps.append(f"Q3 (75th percentile): {q3}")

            else:
                # Complete statistical summary
                result = StatisticalResult(
                    data=data,
                    mean=np.mean(arr),
                    median=np.median(arr),
                    mode=[],
                    variance=np.var(arr),
                    std_dev=np.std(arr),
                    min_val=np.min(arr),
                    max_val=np.max(arr),
                    quartiles={
                        "Q1": np.percentile(arr, 25),
                        "Q2": np.percentile(arr, 50),
                        "Q3": np.percentile(arr, 75),
                    },
                    outliers=[],
                )

                # Calculate mode
                from collections import Counter

                counter = Counter(data)
                max_count = max(counter.values())
                result.mode = [k for k, v in counter.items() if v == max_count]

                # Detect outliers (IQR method)
                iqr = result.quartiles["Q3"] - result.quartiles["Q1"]
                lower_bound = result.quartiles["Q1"] - 1.5 * iqr
                upper_bound = result.quartiles["Q3"] + 1.5 * iqr
                result.outliers = [
                    x for x in data if x < lower_bound or x > upper_bound
                ]

                steps.append(
                    f"Statistical summary computed for {len(data)} data points"
                )

            solution = MathSolution(
                problem=problem,
                solution=result,
                steps=steps if show_steps else [],
                domain=MathDomain.STATISTICS,
                problem_type=problem_type,
                confidence=0.95,
            )

            return solution

        else:
            raise Exception("NumPy required for statistics calculations")

    # ============
    # Number Theory Solver
    # ============

    async def _solve_number_theory(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Solve number theory problems"""
        steps = []
        result = None

        try:
            numbers = re.findall(r"\d+", problem)

            if "prime" in problem.lower():
                if numbers:
                    n = int(numbers[0])
                    result = self._is_prime(n)
                    steps.append(f"Testing if {n} is prime")
                    steps.append(
                        f"Checking divisors up to sqrt({n}) = {int(math.sqrt(n))}"
                    )
                    steps.append(f"Result: {n} is {'prime' if result else 'composite'}")
                else:
                    # Generate primes
                    if "first" in problem.lower():
                        match = re.search(r"first\s+(\d+)", problem.lower())
                        if match:
                            count = int(match.group(1))
                            result = self._generate_primes(count)
                            steps.append(f"Generated first {count} prime numbers")

            elif (
                "gcd" in problem.lower() or "greatest common divisor" in problem.lower()
            ):
                if len(numbers) >= 2:
                    a, b = int(numbers[0]), int(numbers[1])
                    result = math.gcd(a, b)
                    steps.append(f"GCD({a}, {b}) = {result}")
                    steps.append("Using Euclidean algorithm")

            elif "lcm" in problem.lower():
                if len(numbers) >= 2:
                    a, b = int(numbers[0]), int(numbers[1])
                    result = abs(a * b) // math.gcd(a, b)
                    steps.append(f"LCM({a}, {b}) = {result}")

            elif "mod" in problem.lower() or "modulo" in problem.lower():
                if len(numbers) >= 2:
                    a, b = int(numbers[0]), int(numbers[1])
                    result = a % b
                    steps.append(f"{a} mod {b} = {result}")

            elif "factorial" in problem.lower():
                if numbers:
                    n = int(numbers[0])
                    result = math.factorial(n)
                    steps.append(f"{n}! = {result}")

            else:
                # Default: analyze number
                if numbers:
                    n = int(numbers[0])
                    result = {
                        "number": n,
                        "is_prime": self._is_prime(n),
                        "factors": self._factorize(n),
                        "is_perfect_square": int(math.sqrt(n)) ** 2 == n,
                        "digit_sum": sum(int(d) for d in str(n)),
                    }
                    steps.append(f"Analysis of {n}")

            solution = MathSolution(
                problem=problem,
                solution=result,
                steps=steps if show_steps else [],
                domain=MathDomain.NUMBER_THEORY,
                problem_type=problem_type,
                confidence=0.95,
            )

            return solution

        except Exception as e:
            raise Exception(f"Number theory solving failed: {str(e)}")

    def _is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _generate_primes(self, count: int) -> List[int]:
        """Generate first n primes"""
        primes = []
        candidate = 2
        while len(primes) < count:
            if self._is_prime(candidate):
                primes.append(candidate)
            candidate += 1
        return primes

    def _factorize(self, n: int) -> List[int]:
        """Factorize number into prime factors"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    # ============
    # Geometry Solver
    # ============

    async def _solve_geometry(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Solve geometry problems"""
        steps = []

        numbers = re.findall(r"\d+\.?\d*", problem)

        if "circle" in problem.lower():
            if "area" in problem.lower() and numbers:
                radius = float(numbers[0])
                area = math.pi * radius**2
                steps.append(f"Area of circle = π × r² = π × {radius}² = {area:.4f}")
                result = area

            elif "circumference" in problem.lower() and numbers:
                radius = float(numbers[0])
                circumference = 2 * math.pi * radius
                steps.append(
                    f"Circumference = 2πr = 2π × {radius} = {circumference:.4f}"
                )
                result = circumference

        elif "triangle" in problem.lower():
            if len(numbers) >= 2:
                if "area" in problem.lower():
                    base = float(numbers[0])
                    height = float(numbers[1]) if len(numbers) > 1 else None
                    if height:
                        area = 0.5 * base * height
                        steps.append(
                            f"Area of triangle = ½ × base × height = ½ × {base} × {height} = {area}"
                        )
                        result = area

                elif (
                    "pythagorean" in problem.lower() or "hypotenuse" in problem.lower()
                ):
                    a, b = float(numbers[0]), float(numbers[1])
                    c = math.sqrt(a**2 + b**2)
                    steps.append(
                        f"Hypotenuse = √(a² + b²) = √({a}² + {b}²) = √{a**2 + b**2} = {c:.4f}"
                    )
                    result = c

        elif "rectangle" in problem.lower() or "square" in problem.lower():
            if len(numbers) >= 2:
                length = float(numbers[0])
                width = float(numbers[1]) if len(numbers) > 1 else length

                if "area" in problem.lower():
                    area = length * width
                    steps.append(f"Area = length × width = {length} × {width} = {area}")
                    result = area

                elif "perimeter" in problem.lower():
                    perimeter = 2 * (length + width)
                    steps.append(
                        f"Perimeter = 2(length + width) = 2({length} + {width}) = {perimeter}"
                    )
                    result = perimeter

        else:
            raise Exception("Geometry problem not recognized")

        solution = MathSolution(
            problem=problem,
            solution=result,
            steps=steps if show_steps else [],
            domain=MathDomain.GEOMETRY,
            problem_type=problem_type,
            confidence=0.95,
            numerical_value=result if isinstance(result, (int, float)) else None,
        )

        return solution

    # ============
    # Trigonometry Solver
    # ============

    async def _solve_trigonometry(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Solve trigonometry problems"""
        steps = []

        if not self.use_sympy:
            # Numerical evaluation
            numbers = re.findall(r"\d+\.?\d*", problem)

            if "sin" in problem.lower():
                angle = float(numbers[0]) if numbers else 0
                if "deg" in problem.lower():
                    angle = math.radians(angle)
                result = math.sin(angle)
                steps.append(f"sin({angle:.4f} rad) = {result:.6f}")

            elif "cos" in problem.lower():
                angle = float(numbers[0]) if numbers else 0
                if "deg" in problem.lower():
                    angle = math.radians(angle)
                result = math.cos(angle)
                steps.append(f"cos({angle:.4f} rad) = {result:.6f}")

            elif "tan" in problem.lower():
                angle = float(numbers[0]) if numbers else 0
                if "deg" in problem.lower():
                    angle = math.radians(angle)
                result = math.tan(angle)
                steps.append(f"tan({angle:.4f} rad) = {result:.6f}")

            else:
                raise Exception("Trigonometric function not recognized")

        else:
            # Symbolic evaluation
            x = sp.Symbol("x")
            expr = parse_expr(problem)
            result = sp.simplify(expr)
            steps.append(f"Simplified: {sp.latex(result)}")

        solution = MathSolution(
            problem=problem,
            solution=result,
            steps=steps if show_steps else [],
            domain=MathDomain.TRIGONOMETRY,
            problem_type=problem_type,
            confidence=0.95,
            numerical_value=float(result) if isinstance(result, (int, float)) else None,
        )

        return solution

    # ============
    # Discrete Math Solver
    # ============

    async def _solve_discrete(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Solve discrete mathematics problems"""
        steps = []

        numbers = re.findall(r"\d+", problem)

        if "combination" in problem.lower() or "choose" in problem.lower():
            if len(numbers) >= 2:
                n, k = int(numbers[0]), int(numbers[1])
                from math import comb

                result = comb(n, k)
                steps.append(f"C({n}, {k}) = {n}! / ({k}! × {n-k}!) = {result}")

        elif "permutation" in problem.lower():
            if len(numbers) >= 2:
                n, k = int(numbers[0]), int(numbers[1])
                from math import perm

                result = perm(n, k)
                steps.append(f"P({n}, {k}) = {n}! / {n-k}! = {result}")

        elif "factorial" in problem.lower():
            if numbers:
                n = int(numbers[0])
                result = math.factorial(n)
                steps.append(f"{n}! = {result}")

        else:
            raise Exception("Discrete math problem not recognized")

        solution = MathSolution(
            problem=problem,
            solution=result,
            steps=steps if show_steps else [],
            domain=MathDomain.DISCRETE_MATH,
            problem_type=problem_type,
            confidence=0.95,
            numerical_value=result,
        )

        return solution

    # ============
    # Numerical Solver (Fallback)
    # ============

    async def _solve_numerical(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Fallback numerical solver"""
        steps = []

        # Try to evaluate numerically
        try:
            # Replace constants
            expr = problem
            for name, value in self.constants.items():
                expr = expr.replace(name, str(value))

            # Evaluate safely
            namespace = {"__builtins__": {}, "math": math, "cmath": cmath}
            result = eval(expr, namespace)

            steps.append(f"Numerical evaluation: {expr} = {result}")

            solution = MathSolution(
                problem=problem,
                solution=result,
                steps=steps if show_steps else [],
                domain=MathDomain.ARITHMETIC,
                problem_type=problem_type,
                confidence=0.8,
                numerical_value=(
                    float(result) if isinstance(result, (int, float)) else None
                ),
            )

            return solution

        except Exception as e:
            raise Exception(f"Numerical evaluation failed: {str(e)}")

    async def _solve_generic(
        self, problem: str, problem_type: ProblemType, variables: Dict, show_steps: bool
    ) -> MathSolution:
        """Generic solver for unclassified problems"""
        return await self._solve_numerical(problem, problem_type, variables, show_steps)

    # ============
    # Utility Methods
    # ============

    def _format_solution(self, solution: MathSolution, start_time: datetime) -> Dict:
        """Format solution for output"""
        return {
            "success": True,
            "problem": solution.problem,
            "solution": (
                solution.solution
                if not isinstance(solution.solution, (sp.Basic, sp.MatrixBase))
                else str(solution.solution)
            ),
            "steps": solution.steps,
            "domain": solution.domain.value,
            "problem_type": solution.problem_type.value,
            "confidence": solution.confidence,
            "execution_time": solution.execution_time,
            "latex": solution.latex,
            "numerical_value": solution.numerical_value,
            "timestamp": datetime.now().isoformat(),
        }

    async def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression numerically

        Args:
            expression: Mathematical expression

        Returns:
            Dictionary with evaluation result
        """
        try:
            # Replace constants
            expr = expression
            for name, value in self.constants.items():
                expr = expr.replace(name, str(value))

            # Evaluate
            namespace = {"__builtins__": {}, "math": math, "cmath": cmath}
            result = eval(expr, namespace)

            return {
                "success": True,
                "expression": expression,
                "result": result,
                "type": type(result).__name__,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "expression": expression}

    async def generate_sequence(
        self, formula: str, n_terms: int = 10
    ) -> Dict[str, Any]:
        """
        Generate sequence terms from formula

        Args:
            formula: Formula with 'n' variable
            n_terms: Number of terms to generate

        Returns:
            Dictionary with sequence
        """
        terms = []

        try:
            for n in range(1, n_terms + 1):
                expr = formula.replace("n", str(n))
                namespace = {"__builtins__": {}, "math": math}
                term = eval(expr, namespace)
                terms.append(term)

            return {
                "success": True,
                "formula": formula,
                "terms": terms,
                "n_terms": n_terms,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "formula": formula}

    def get_history(self, limit: int = None) -> List[Dict]:
        """Get solution history"""
        history = self.solution_history
        if limit:
            history = history[-limit:]

        return [
            {
                "problem": h.problem[:100],
                "domain": h.domain.value,
                "problem_type": h.problem_type.value,
                "execution_time": h.execution_time,
                "timestamp": datetime.now().isoformat(),
            }
            for h in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        success_rate = (
            (self.stats["solved_problems"] / self.stats["total_problems"] * 100)
            if self.stats["total_problems"] > 0
            else 0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "history_size": len(self.solution_history),
            "cache_size": len(self.solution_cache),
            "sympy_available": SYMPY_AVAILABLE,
            "numpy_available": NP_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE,
        }

    def clear_history(self):
        """Clear solution history"""
        self.solution_history.clear()
        self.logger.info("Solution history cleared")

    def clear_cache(self):
        """Clear solution cache"""
        self.solution_cache.clear()
        self.logger.info("Solution cache cleared")


# Integration wrapper for EDIATH
class MathReasoningAgentWrapper:
    """
    Wrapper class to integrate MathReasoningAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.math_agent = MathReasoningAgent(config)
        self.agent_type = "math_reasoning"
        self.capabilities = [
            "solve_problem",
            "evaluate_expression",
            "generate_sequence",
            "statistical_analysis",
            "matrix_operations",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a math request

        Request format:
        {
            'operation': 'solve|evaluate|sequence|stats|history',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "solve":
            domain = request.get("domain")
            problem_type = request.get("problem_type")

            return await self.math_agent.solve(
                problem=request.get("problem"),
                domain=MathDomain(domain) if domain else None,
                problem_type=ProblemType(problem_type) if problem_type else None,
                variables=request.get("variables"),
                show_steps=request.get("show_steps", True),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "evaluate":
            return await self.math_agent.evaluate_expression(
                expression=request.get("expression")
            )

        elif operation == "sequence":
            return await self.math_agent.generate_sequence(
                formula=request.get("formula"), n_terms=request.get("n_terms", 10)
            )

        elif operation == "history":
            return {
                "success": True,
                "history": self.math_agent.get_history(limit=request.get("limit")),
            }

        elif operation == "stats":
            return self.math_agent.get_stats()

        elif operation == "clear_history":
            self.math_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        elif operation == "clear_cache":
            self.math_agent.clear_cache()
            return {"success": True, "message": "Cache cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "MathReasoningAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.math_agent.get_stats(),
            "domains": [d.value for d in MathDomain],
            "sympy_available": SYMPY_AVAILABLE,
        }


# Example usage and testing
async def test_math_agent():
    """Test the math reasoning agent functionality"""

    # Initialize agent
    agent = MathReasoningAgent()

    print("=== Math Reasoning Agent Test ===\n")

    # Test algebra - simplification
    print("1. Algebra - Simplification...")
    result = await agent.solve("x**2 + 2*x + 1", domain=MathDomain.ALGEBRA)
    if result["success"]:
        print(f"   Problem: {result['problem']}")
        print(f"   Solution: {result['solution']}")
        print(f"   Steps: {len(result['steps'])} steps")

    # Test equation solving
    print("\n2. Algebra - Equation Solving...")
    result = await agent.solve("x**2 - 5*x + 6 = 0", domain=MathDomain.ALGEBRA)
    if result["success"]:
        print(f"   Equation: {result['problem']}")
        print(f"   Solutions: {result['solution']}")

    # Test calculus - derivative
    print("\n3. Calculus - Derivative...")
    result = await agent.solve("x**3 * sin(x)", domain=MathDomain.CALCULUS)
    if result["success"]:
        print(f"   Function: {result['problem']}")
        print(f"   Derivative: {result['solution']}")
        if result.get("latex"):
            print(f"   LaTeX: {result['latex']}")

    # Test calculus - integral
    print("\n4. Calculus - Integral...")
    result = await agent.solve("integrate x**2 * exp(x)", domain=MathDomain.CALCULUS)
    if result["success"]:
        print(f"   Integral: {result['solution']}")

    # Test statistics
    print("\n5. Statistics - Data Analysis...")
    result = await agent.solve(
        "Statistics of [1,2,3,4,5,6,7,8,9,10]", domain=MathDomain.STATISTICS
    )
    if result["success"]:
        sol = result["solution"]
        print(f"   Mean: {sol.get('mean', 'N/A')}")
        print(f"   Median: {sol.get('median', 'N/A')}")
        print(f"   Std Dev: {sol.get('std_dev', 'N/A'):.4f}")

    # Test number theory
    print("\n6. Number Theory - Prime Numbers...")
    result = await agent.solve("Is 97 prime?", domain=MathDomain.NUMBER_THEORY)
    if result["success"]:
        print(f"   Result: {result['solution']}")

    result = await agent.solve(
        "First 10 prime numbers", domain=MathDomain.NUMBER_THEORY
    )
    if result["success"]:
        print(f"   First 10 primes: {result['solution']}")

    # Test geometry
    print("\n7. Geometry - Circle Area...")
    result = await agent.solve("Circle area with radius 5", domain=MathDomain.GEOMETRY)
    if result["success"]:
        print(f"   Area: {result['numerical_value']:.4f}")

    # Test trigonometry
    print("\n8. Trigonometry...")
    result = await agent.solve("sin(30 degrees)", domain=MathDomain.TRIGONOMETRY)
    if result["success"]:
        print(f"   sin(30°): {result['numerical_value']:.6f}")

    # Test discrete math
    print("\n9. Discrete Math - Combinations...")
    result = await agent.solve(
        "Combination of 10 choose 3", domain=MathDomain.DISCRETE_MATH
    )
    if result["success"]:
        print(f"   C(10,3) = {result['numerical_value']}")

    # Test expression evaluation
    print("\n10. Expression Evaluation...")
    result = await agent.evaluate_expression("sqrt(2) * pi")
    if result["success"]:
        print(f"   √2 × π = {result['result']:.6f}")

    # Test sequence generation
    print("\n11. Sequence Generation...")
    result = await agent.generate_sequence("n**2", n_terms=5)
    if result["success"]:
        print(f"   Sequence: {result['terms']}")

    # Get statistics
    print("\n12. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total problems: {stats['total_problems']}")
    print(f"   Solved: {stats['solved_problems']}")
    print(f"   Failed: {stats['failed_problems']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Average time: {stats['average_time']:.3f}s")
    print(f"   By domain: {stats['by_domain']}")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_math_agent())
