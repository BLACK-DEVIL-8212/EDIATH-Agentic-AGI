"""
Advanced Software Builder - AI Code Generation Engine (UPGRADED)
Capabilities: Requirements Analysis, Project Planning, Iterative Development, Testing, Documentation

USAGE EXAMPLE:
    builder = SoftwareBuilder()
    result = await builder.develop_from_requirements(
        "my_web_app",
        "Create a Flask web application that displays weather information for a given city"
    )
    print(f"Project created at: {result['path']}")
"""

from typing import Dict, Any, List
from pathlib import Path
import json

from ..utils.logger import logger

# Try to import LLMEngine, fallback to mock if not available
try:
    from ..brain.llm_engine import LLMEngine

    LLM_AVAILABLE = True
except ImportError:
    logger.warning("LLMEngine not available, using mock implementation")
    LLM_AVAILABLE = False

    class LLMEngine:
        """Mock LLM Engine for testing when real LLM is not available."""

        async def generate(self, prompt: str) -> str:
            # Return mock responses based on prompt content
            if "analyze these software requirements" in prompt.lower():
                return json.dumps(
                    {
                        "summary": "Command-line calculator application",
                        "type": "tool",
                        "complexity": "simple",
                        "technologies": ["python"],
                        "components": [
                            {
                                "name": "calculator",
                                "description": "Basic arithmetic operations",
                                "files": ["calculator.py"],
                                "dependencies": [],
                            }
                        ],
                        "features": ["add", "subtract", "multiply", "divide"],
                        "estimated_time": "2 hours",
                    }
                )
            elif "design the architecture" in prompt.lower():
                return json.dumps(
                    {
                        "architecture": "modular",
                        "patterns": ["procedural"],
                        "structure": {
                            "folders": ["src", "tests"],
                            "main_files": ["main.py"],
                            "modules": ["calculator.py"],
                        },
                        "data_flow": "Input -> Operation -> Result",
                        "design_decisions": ["Simple modular design"],
                    }
                )
            else:
                return 'def hello_world():\n    """A simple hello world function."""\n    print("Hello, World!")\n    return "Hello, World!"'


class SoftwareBuilder:
    def __init__(self, base_dir="generated_projects"):
        self.projects = {}
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        self.llm = LLMEngine()

        # 🔥 ENHANCED STATS
        self.projects_built = 0
        self.tasks_completed = 0
        self.files_generated = 0
        self.requirements_processed = 0
        self.tests_passed = 0
        self.execution_history: List[Dict[str, Any]] = []
        self.development_sessions: List[Dict[str, Any]] = []

    # ------------------------
    # 📋 ANALYZE REQUIREMENTS
    # ------------------------
    async def analyze_requirements(self, requirements_text: str) -> Dict[str, Any]:
        """Analyze user requirements and break them down into structured components."""
        prompt = f"""
Analyze these software requirements and provide a structured breakdown:

Requirements:
{requirements_text}

Return JSON with:
{{
  "summary": "Brief project summary",
  "type": "web|desktop|mobile|api|library|tool",
  "complexity": "simple|medium|complex",
  "technologies": ["python", "flask", "sqlite", etc.],
  "components": [
    {{
      "name": "component_name",
      "description": "what it does",
      "files": ["file1.py", "file2.py"],
      "dependencies": ["library1", "library2"]
    }}
  ],
  "features": ["feature1", "feature2"],
  "estimated_time": "hours/days"
}}
"""

        response = await self.generate_code(prompt)
        try:
            analysis = json.loads(response)
            self.requirements_processed += 1
            logger.info(
                f"📋 Requirements analyzed: {analysis.get('summary', 'Unknown project')}"
            )
            return analysis
        except Exception as e:
            logger.error(f"Failed to parse requirements analysis: {e}")
            return {
                "summary": requirements_text[:100] + "...",
                "type": "unknown",
                "complexity": "medium",
                "technologies": ["python"],
                "components": [],
                "features": [],
                "estimated_time": "unknown",
            }

    # ------------------------
    # 🏗️ DESIGN ARCHITECTURE
    # ------------------------
    async def design_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Design the software architecture based on requirements analysis."""
        prompt = f"""
Design the architecture for this software project:

Analysis: {json.dumps(analysis, indent=2)}

Return JSON with:
{{
  "architecture": "monolithic|microservices|modular",
  "patterns": ["pattern1", "pattern2"],
  "structure": {{
    "folders": ["src/", "tests/", "docs/"],
    "main_files": ["main.py", "config.py"],
    "modules": ["module1.py", "module2.py"]
  }},
  "data_flow": "description of data flow",
  "design_decisions": ["decision1", "decision2"]
}}
"""

        response = await self.generate_code(prompt)
        try:
            return json.loads(response)
        except:
            return {
                "architecture": "modular",
                "patterns": ["MVC"],
                "structure": {
                    "folders": ["src", "tests"],
                    "main_files": ["main.py"],
                    "modules": [],
                },
                "data_flow": "Standard input/output flow",
                "design_decisions": ["Modular design for maintainability"],
            }

    def create_project(self, name: str, architecture: Dict[str, Any] = None):
        project_path = self.base_dir / name
        project_path.mkdir(exist_ok=True)

        # 🔥 AUTO STRUCTURE BASED ON ARCHITECTURE
        if architecture:
            structure = architecture.get("structure", {})
            for folder in structure.get("folders", ["src", "tests", "docs"]):
                (project_path / folder).mkdir(exist_ok=True)

            # Create __init__.py files for Python packages
            for folder in ["src", "tests"]:
                init_file = project_path / folder / "__init__.py"
                if not init_file.exists():
                    with open(init_file, "w") as f:
                        f.write('"""Package initialization."""\n')
        else:
            # Default structure
            (project_path / "src").mkdir(exist_ok=True)
            (project_path / "tests").mkdir(exist_ok=True)

        self.proj
