"""
Advanced Human Browser - Ultimate Edition (AI + Human Behavior Simulation)
✔ Realistic mouse movements (bezier curves) & scrolling (with inertia)
✔ Typing speed variation (per-character with cognitive pauses)
✔ Tab management (multiple tabs, switching, closing)
✔ Session persistence (cookies, localStorage, IndexedDB)
✔ AI-driven link selection (using EDIATH's LLM with relevance scoring)
✔ Human-like hesitation, reading time, back/forward navigation
✔ Eye movement simulation (scanning patterns)
✔ Random micro-expressions (hesitation, pauses, corrections)
✔ Behavioral profiles (casual, researcher, shopper, bot-detection-avoidance)
✔ Mouse trajectory recording & playback
✔ Form filling with realistic mistakes & corrections
✔ Download handling
✔ Screenshot with eye-tracking heatmaps
✔ Full async support
✔ Memory integration with EDIATH
"""

import asyncio
import random
import re
import json
import math
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..utils.logger import logger
from .chrome_controller import create_chrome_controller

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy.interpolate import CubicSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class UserProfile(Enum):
    """User behavioral profiles"""
    CASUAL = "casual"           # Slow, random, many pauses
    RESEARCHER = "researcher"   # Methodical, reads thoroughly, few clicks
    SHOPPER = "shopper"         # Fast, price-focused, compares products
    SOCIAL = "social"           # Scrolls fast, clicks many links, short attention
    PROFESSIONAL = "professional"  # Efficient, minimal movements, goal-oriented
    AVOID_DETECTION = "avoid_detection"  # Mimics human randomness to avoid bot detection


class EyeMovement(Enum):
    """Eye movement patterns"""
    FOCUSED = "focused"      # Concentrated on specific areas
    SCANNING = "scanning"    # Fast, across the page
    READING = "reading"      # Left-to-right, top-to-bottom
    RANDOM = "random"        # Random saccades


@dataclass
class MouseTrajectory:
    """Mouse movement trajectory"""
    points: List[Tuple[int, int]]
    timestamps: List[float]
    duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": self.points,
            "timestamps": self.timestamps,
            "duration": self.duration
        }


@dataclass
class InteractionRecord:
    """Record of human interaction"""
    action: str
    details: Dict[str, Any]
    timestamp: datetime
    url: str
    duration_ms: float
    mouse_trajectory: Optional[MouseTrajectory] = None


class HumanBehavior:
    """Advanced human-like timing and probability distributions"""
    
    @staticmethod
    def uniform(min_sec: float = 0.3, max_sec: float = 1.5) -> float:
        """Uniform distribution"""
        return random.uniform(min_sec, max_sec)
    
    @staticmethod
    def normal(mean: float = 1.0, std: float = 0.3) -> float:
        """Normal (Gaussian) distribution"""
        return max(0.05, random.gauss(mean, std))
    
    @staticmethod
    def exponential(scale: float = 1.0) -> float:
        """Exponential distribution for rare events"""
        return random.expovariate(1.0 / scale)
    
    @staticmethod
    def beta(alpha: float = 2.0, beta: float = 5.0) -> float:
        """Beta distribution for bounded random values"""
        return random.betavariate(alpha, beta)
    
    @staticmethod
    def scroll_amount(profile: UserProfile = UserProfile.CASUAL) -> int:
        """Human-like scroll amounts with inertia"""
        if profile == UserProfile.CASUAL:
            if random.random() < 0.7:
                return random.randint(50, 200)
            else:
                return random.randint(300, 800)
        elif profile == UserProfile.RESEARCHER:
            return random.randint(100, 250)
        elif profile == UserProfile.SOCIAL:
            return random.randint(200, 600)
        elif profile == UserProfile.PROFESSIONAL:
            return random.randint(80, 300)
        else:  # AVOID_DETECTION
            return random.choice([50, 100, 150, 250, 400, 600])
    
    @staticmethod
    def scroll_inertia(prev_amount: int) -> int:
        """Simulate scrolling inertia (continued movement)"""
        if random.random() < 0.3:
            # Continue scrolling in same direction
            return int(prev_amount * random.uniform(0.5, 0.9))
        elif random.random() < 0.1:
            # Overshoot and correct
            return -int(prev_amount * random.uniform(0.2, 0.5))
        return 0
    
    @staticmethod
    def mouse_offset() -> Tuple[int, int]:
        """Random offset for mouse clicks (avoid pixel-perfect clicks)"""
        return (random.randint(-8, 8), random.randint(-8, 8))
    
    @staticmethod
    def hesitation_before_action(profile: UserProfile = UserProfile.CASUAL) -> float:
        """Hesitation before clicking/interacting"""
        if profile == UserProfile.CASUAL:
            return random.uniform(0.3, 1.2)
        elif profile == UserProfile.RESEARCHER:
            return random.uniform(0.5, 2.0)
        elif profile == UserProfile.SHOPPER:
            return random.uniform(0.1, 0.5)
        elif profile == UserProfile.PROFESSIONAL:
            return random.uniform(0.2, 0.8)
        else:  # AVOID_DETECTION
            return random.uniform(0.2, 1.5)
    
    @staticmethod
    def should_click(profile: UserProfile = UserProfile.CASUAL) -> bool:
        """Probability of clicking on a given element"""
        probs = {
            UserProfile.CASUAL: 0.35,
            UserProfile.RESEARCHER: 0.25,
            UserProfile.SHOPPER: 0.45,
            UserProfile.SOCIAL: 0.55,
            UserProfile.PROFESSIONAL: 0.30,
            UserProfile.AVOID_DETECTION: 0.40
        }
        return random.random() < probs.get(profile, 0.4)
    
    @staticmethod
    def should_scroll(profile: UserProfile = UserProfile.CASUAL) -> bool:
        """Probability of scrolling"""
        probs = {
            UserProfile.CASUAL: 0.65,
            UserProfile.RESEARCHER: 0.80,
            UserProfile.SHOPPER: 0.70,
            UserProfile.SOCIAL: 0.85,
            UserProfile.PROFESSIONAL: 0.60,
            UserProfile.AVOID_DETECTION: 0.75
        }
        return random.random() < probs.get(profile, 0.7)
    
    @staticmethod
    def should_hesitate(profile: UserProfile = UserProfile.CASUAL) -> bool:
        """Probability of hesitating before action"""
        probs = {
            UserProfile.CASUAL: 0.4,
            UserProfile.RESEARCHER: 0.3,
            UserProfile.SHOPPER: 0.2,
            UserProfile.PROFESSIONAL: 0.15,
            UserProfile.AVOID_DETECTION: 0.35
        }
        return random.random() < probs.get(profile, 0.3)
    
    @staticmethod
    def typing_delay(profile: UserProfile = UserProfile.CASUAL, char: str = "") -> float:
        """Realistic typing delays with cognitive pauses"""
        base_delay = 0.05
        
        # Cognitive pauses at certain characters
        if char in ['.', '!', '?']:
            base_delay += random.uniform(0.1, 0.3)  # End of sentence
        elif char in [',', ';']:
            base_delay += random.uniform(0.05, 0.15)  # Pause at punctuation
        elif char.isspace() and random.random() < 0.3:
            base_delay += random.uniform(0.1, 0.4)  # Random word pause
        
        # Profile-specific typing speeds
        if profile == UserProfile.CASUAL:
            return base_delay + random.expovariate(15)
        elif profile == UserProfile.RESEARCHER:
            return base_delay + random.expovariate(12)  # Slower, more deliberate
        elif profile == UserProfile.PROFESSIONAL:
            return base_delay + random.expovariate(20)  # Faster
        else:
            return base_delay + random.expovariate(15)
    
    @staticmethod
    def make_mistake(profile: UserProfile = UserProfile.CASUAL) -> bool:
        """Probability of making a typing mistake"""
        probs = {
            UserProfile.CASUAL: 0.08,
            UserProfile.RESEARCHER: 0.03,
            UserProfile.SHOPPER: 0.10,
            UserProfile.SOCIAL: 0.12,
            UserProfile.PROFESSIONAL: 0.02,
            UserProfile.AVOID_DETECTION: 0.07
        }
        return random.random() < probs.get(profile, 0.07)
    
    @staticmethod
    def reading_time(text_length: int, profile: UserProfile = UserProfile.CASUAL) -> float:
        """Human reading time based on text length and profile"""
        # Average reading speed: 200-300 words per minute
        words = text_length / 5
        base_seconds = (words / 250) * 60
        
        # Profile multipliers
        multipliers = {
            UserProfile.CASUAL: random.uniform(0.8, 1.2),
            UserProfile.RESEARCHER: random.uniform(1.0, 1.5),
            UserProfile.SHOPPER: random.uniform(0.5, 0.8),
            UserProfile.SOCIAL: random.uniform(0.4, 0.7),
            UserProfile.PROFESSIONAL: random.uniform(0.7, 1.0),
            UserProfile.AVOID_DETECTION: random.uniform(0.8, 1.5)
        }
        
        reading_time = base_seconds * multipliers.get(profile, 1.0)
        return min(30, max(0.5, reading_time))
    
    @staticmethod
    def eye_saccade_duration() -> float:
        """Duration of eye movement between fixations"""
        return random.uniform(0.03, 0.07)
    
    @staticmethod
    def fixation_duration() -> float:
        """Duration of eye fixation on a point"""
        return random.uniform(0.2, 0.5)
    
    @staticmethod
    def should_continue(profile: UserProfile = UserProfile.CASUAL) -> bool:
        """Should continue browsing session"""
        if profile == UserProfile.SOCIAL:
            return random.random() < 0.9  # High engagement
        elif profile == UserProfile.RESEARCHER:
            return random.random() < 0.95
        else:
            return random.random() < 0.8


class MouseMovementSimulator:
    """Advanced mouse movement simulation with bezier curves"""
    
    @staticmethod
    def bezier_curve(p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int], t: float) -> Tuple[int, int]:
        """Quadratic bezier curve"""
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        return (int(x), int(y))
    
    @staticmethod
    def generate_trajectory(
        start: Tuple[int, int],
        end: Tuple[int, int],
        duration_ms: float = 300,
        control_points: int = 2
    ) -> MouseTrajectory:
        """Generate realistic mouse trajectory"""
        points = []
        timestamps = []
        
        # Generate random control points for natural curve
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        cp1 = (
            int(mid_x + random.uniform(-100, 100)),
            int(mid_y + random.uniform(-50, 50))
        )
        
        # Number of points based on duration
        num_points = max(10, int(duration_ms / 15))
        
        for i in range(num_points + 1):
            t = i / num_points
            # Easing function for human-like acceleration/deceleration
            eased_t = t ** 2 / (t ** 2 + (1 - t) ** 2) if t > 0 else 0
            
            point = MouseMovementSimulator.bezier_curve(start, cp1, end, eased_t)
            
            # Add small random jitter
            point = (
                point[0] + random.randint(-2, 2),
                point[1] + random.randint(-2, 2)
            )
            
            points.append(point)
            timestamps.append(t * duration_ms / 1000)
        
        return MouseTrajectory(
            points=points,
            timestamps=timestamps,
            duration=duration_ms / 1000
        )
    
    @staticmethod
    async def execute_mouse_movement(
        page,
        start: Tuple[int, int],
        end: Tuple[int, int],
        duration_ms: float = 300
    ):
        """Execute smooth mouse movement"""
        trajectory = MouseMovementSimulator.generate_trajectory(start, end, duration_ms)
        
        for i, point in enumerate(trajectory.points):
            # Move mouse to position
            await page.mouse.move(point[0], point[1])
            
            # Wait for next step
            if i < len(trajectory.points) - 1:
                step_duration = trajectory.timestamps[i + 1] - trajectory.timestamps[i]
                await asyncio.sleep(step_duration)


class HumanBrowser:
    """
    Ultimate Human Browser with maximum human behavior simulation
    """
    
    def __init__(
        self,
        headless: bool = False,
        user_data_dir: Optional[str] = None,
        session_persistence: bool = True,
        use_ai_for_links: bool = True,
        profile: UserProfile = UserProfile.CASUAL,
        eye_movement: EyeMovement = EyeMovement.READING,
        record_trajectories: bool = True,
        max_concurrent_tabs: int = 5,
        screenshot_on_action: bool = False,
    ):
        """
        Initialize Human Browser
        
        Args:
            headless: Run browser in headless mode (Note: ChromeController uses headless parameter via config)
            user_data_dir: Directory for persistent session data
            session_persistence: Save/load session between runs
            use_ai_for_links: Use LLM for intelligent link selection
            profile: User behavioral profile
            eye_movement: Eye movement pattern
            record_trajectories: Record mouse movement trajectories
            max_concurrent_tabs: Maximum number of tabs to keep open
            screenshot_on_action: Take screenshot on each action
        """
        # Store parameters for later use
        self.headless = headless
        self.browser = create_chrome_controller(headless=headless)
        
        self.behavior = HumanBehavior()
        self.profile = profile
        self.eye_movement = eye_movement
        self.use_ai_for_links = use_ai_for_links
        self.session_persistence = session_persistence
        self.record_trajectories = record_trajectories
        self.max_concurrent_tabs = max_concurrent_tabs
        self.screenshot_on_action = screenshot_on_action
        self.user_data_dir = user_data_dir or str(Path("data/human_browser_profile"))
        
        # Session state
        self.session_start = datetime.now()
        self.last_action_time = datetime.now()
        self.pages_visited = 0
        self.current_url = ""
        self.interactions: List[InteractionRecord] = []
        self.page_history: List[str] = []
        self.history_index = -1
        
        # Tab management
        self.tabs: Dict[int, Dict[str, Any]] = {}
        self.current_tab_id = 0
        self.tab_counter = 0
        
        # Mouse trajectory recording
        self.trajectories: List[MouseTrajectory] = []
        
        # Eye tracking simulation
        self.eye_fixations: List[Dict[str, Any]] = []
        
        # AI decision engine
        self._llm = None
        # The shared LLM can be injected later. Do not load it during browser
        # construction because that blocks UI startup.
        
        # Download tracking
        self.downloads: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.interaction_times: List[float] = []
        
        # Patterns (learned from behavior)
        self.learned_patterns: Dict[str, Any] = {}
        
        logger.info(f"🧑‍💻 Human browser initialized with profile: {profile.value}")

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm

        if not self.use_ai_for_links:
            return None

        try:
            from ..brain.llm_engine import LLMEngine

            self._llm = LLMEngine()
        except Exception as e:
            logger.warning(f"LLM not available for AI link selection: {e}")
            self._llm = None
        return self._llm
    
    # ==================== SESSION PERSISTENCE ====================
    
    async def _save_session(self):
        """Save browser session to disk"""
        if not self.session_persistence:
            return
        
        try:
            Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)
            
            session_data = {
                "history": self.page_history,
                "current_url": self.current_url,
                "history_index": self.history_index,
                "interactions": [
                    {
                        "action": i.action,
                        "details": i.details,
                        "timestamp": i.timestamp.isoformat(),
                        "url": i.url,
                        "duration_ms": i.duration_ms
                    }
                    for i in self.interactions[-500:]  # Last 500 interactions
                ],
                "pages_visited": self.pages_visited,
                "session_start": self.session_start.isoformat(),
                "timestamp": datetime.now().isoformat(),
                "profile": self.profile.value,
                "learned_patterns": self.learned_patterns
            }
            
            with open(Path(self.user_data_dir) / "session.json", "w") as f:
                json.dump(session_data, f, indent=2)
            
            # Save cookies and storage via browser context
            if hasattr(self.browser, 'context') and self.browser.context:
                try:
                    storage_path = Path(self.user_data_dir) / "storage.json"
                    await self.browser.context.storage_state(path=storage_path)
                except Exception as e:
                    logger.debug(f"Failed to save storage state: {e}")
            
            logger.debug(f"Session saved to {self.user_data_dir}")
            
        except Exception as e:
            logger.debug(f"Session save failed: {e}")
    
    async def _load_session(self):
        """Load browser session from disk"""
        if not self.session_persistence:
            return
        
        try:
            session_path = Path(self.user_data_dir) / "session.json"
            if session_path.exists():
                with open(session_path, "r") as f:
                    data = json.load(f)
                
                self.page_history = data.get("history", [])
                self.current_url = data.get("current_url", "")
                self.history_index = data.get("history_index", -1)
                self.pages_visited = data.get("pages_visited", 0)
                self.learned_patterns = data.get("learned_patterns", {})
                
                # Restore interactions (without full objects)
                for i in data.get("interactions", []):
                    self.interactions.append(InteractionRecord(
                        action=i["action"],
                        details=i["details"],
                        timestamp=datetime.fromisoformat(i["timestamp"]),
                        url=i["url"],
                        duration_ms=i["duration_ms"]
                    ))
                
                logger.info(f"Loaded session with {len(self.page_history)} pages visited")
            
            # Load storage state
            storage_path = Path(self.user_data_dir) / "storage.json"
            if storage_path.exists() and hasattr(self.browser, 'context') and self.browser.context:
                try:
                    with open(storage_path, "r") as f:
                        storage_state = json.load(f)
                    await self.browser.context.add_cookies(storage_state.get("cookies", []))
                except Exception as e:
                    logger.debug(f"Failed to load storage state: {e}")
                
        except Exception as e:
            logger.debug(f"Session load failed: {e}")
    
    # ==================== START / STOP ====================
    
    async def start(self):
        """Start the human browser"""
        await self.browser.start()
        await self._load_session()
        self.session_start = datetime.now()
        logger.info("🧑‍💻 Human browser started")
        
        # Get current page after start
        if hasattr(self.browser, 'page') and self.browser.page:
            self.current_url = self.browser.page.url
    
    async def stop(self):
        """Stop the human browser and save session"""
        await self._save_session()
        await self.browser.stop()
        logger.info(f"🛑 Human browser stopped. Visited {self.pages_visited} pages")
    
    # ==================== EYE MOVEMENT SIMULATION ====================
    
    async def _simulate_eye_movement(self, target_area: Optional[Tuple[int, int, int, int]] = None):
        """Simulate realistic eye movement patterns"""
        
        if self.eye_movement == EyeMovement.READING:
            # Simulate reading pattern: left to right, top to bottom
            for _ in range(random.randint(3, 8)):
                await asyncio.sleep(self.behavior.fixation_duration())
                saccade = self.behavior.eye_saccade_duration()
                await asyncio.sleep(saccade)
                
        elif self.eye_movement == EyeMovement.SCANNING:
            # Fast scanning of page
            for _ in range(random.randint(5, 15)):
                await asyncio.sleep(self.behavior.fixation_duration() * 0.5)
                await asyncio.sleep(self.behavior.eye_saccade_duration() * 0.3)
                
        elif self.eye_movement == EyeMovement.FOCUSED and target_area:
            # Focus on specific area
            fixations = random.randint(3, 7)
            for _ in range(fixations):
                await asyncio.sleep(self.behavior.fixation_duration() * 1.5)
                # Small saccades within target area
                await asyncio.sleep(self.behavior.eye_saccade_duration())
                
        elif self.eye_movement == EyeMovement.RANDOM:
            # Random eye movements
            for _ in range(random.randint(2, 10)):
                await asyncio.sleep(self.behavior.fixation_duration() * random.uniform(0.5, 2))
                await asyncio.sleep(self.behavior.eye_saccade_duration() * random.uniform(0.5, 1.5))
        
        # Record fixation
        self.eye_fixations.append({
            "timestamp": datetime.now().isoformat(),
            "area": target_area,
            "pattern": self.eye_movement.value
        })
    
    # ==================== MOUSE MOVEMENT ====================
    
    async def _move_mouse_to_element(
        self,
        selector: str,
        offset: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int]]:
        """Simulate realistic mouse movement to element"""
        try:
            if not hasattr(self.browser, 'page') or not self.browser.page:
                return None
                
            # Get element position
            element = await self.browser.page.query_selector(selector)
            if not element:
                return None
            
            box = await element.bounding_box()
            if not box:
                return None
            
            # Calculate target position (with offset)
            target_x = box['x'] + box['width'] / 2
            target_y = box['y'] + box['height'] / 2
            
            if offset:
                target_x += offset[0]
                target_y += offset[1]
            else:
                # Add human-like offset
                offset_x, offset_y = self.behavior.mouse_offset()
                target_x += offset_x
                target_y += offset_y
            
            # Get current mouse position (approximate center of viewport)
            viewport = await self.browser.page.evaluate("window.visualViewport")
            current_x = viewport['width'] // 2
            current_y = viewport['height'] // 2
            
            # Simulate movement
            duration = self.behavior.uniform(0.2, 0.6) * 1000  # ms
            
            if self.record_trajectories:
                trajectory = MouseMovementSimulator.generate_trajectory(
                    (int(current_x), int(current_y)),
                    (int(target_x), int(target_y)),
                    duration
                )
                self.trajectories.append(trajectory)
            
            # Execute movement
            await MouseMovementSimulator.execute_mouse_movement(
                self.browser.page,
                (int(current_x), int(current_y)),
                (int(target_x), int(target_y)),
                duration
            )
            
            return (int(target_x), int(target_y))
            
        except Exception as e:
            logger.debug(f"Mouse movement failed: {e}")
            return None
    
    # ==================== HUMAN NAVIGATION ====================
    
    async def _hesitate(self, action: str = "action"):
        """Simulate human hesitation before action"""
        if self.behavior.should_hesitate(self.profile):
            hesitation_time = self.behavior.hesitation_before_action(self.profile)
            await asyncio.sleep(hesitation_time)
            self._log("hesitation", {"action": action, "duration": hesitation_time})
    
    async def visit(self, url: str, reason: str = "navigation"):
        """Navigate to URL with human-like delays"""
        start_time = datetime.now()
        
        # Hesitation before navigation
        await self._hesitate("navigate")
        
        # Small delay before typing URL
        if reason != "back_forward":
            await asyncio.sleep(self.behavior.uniform(0.2, 0.8))
        
        # Navigate
        await self.browser.navigate(url)
        
        # Update history
        if self.history_index < len(self.page_history) - 1:
            self.page_history = self.page_history[:self.history_index + 1]
        self.page_history.append(url)
        self.history_index = len(self.page_history) - 1
        self.current_url = url
        self.pages_visited += 1
        
        # Simulate reading time
        content = await self.browser.get_content()
        text_length = len(content)
        reading_time = self.behavior.reading_time(text_length, self.profile)
        await asyncio.sleep(reading_time)
        
        # Simulate eye movement scanning
        await self._simulate_eye_movement()
        
        # Record interaction
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._log("visit", {"url": url, "reason": reason, "reading_time": reading_time}, duration_ms)
        
        # Save session
        await self._save_session()
        
        logger.info(f"📄 Visited: {url} (read for {reading_time:.1f}s)")
        return True
    
    # ==================== SCROLLING ====================
    
    async def scroll(self, direction: str = "down", amount: Optional[int] = None):
        """Scroll with human-like behavior and inertia"""
        if not hasattr(self.browser, 'page') or not self.browser.page:
            return
            
        if amount is None:
            amount = self.behavior.scroll_amount(self.profile)
        
        if direction == "up":
            amount = -amount
        
        # Simulate scroll with inertia
        await self.browser.page.mouse.wheel(0, amount)
        self._log("scroll", {"direction": direction, "amount": amount})
        
        # Add inertia continuation
        inertia = self.behavior.scroll_inertia(amount)
        if inertia != 0:
            await asyncio.sleep(self.behavior.uniform(0.1, 0.4))
            await self.browser.page.mouse.wheel(0, inertia)
            self._log("scroll_inertia", {"amount": inertia})
        
        # Small pause after scrolling
        await asyncio.sleep(self.behavior.uniform(0.1, 0.3))
    
    async def scroll_and_read(self, duration: float = None):
        """Scroll and read content naturally"""
        if duration is None:
            duration = self.behavior.uniform(3, 10)
        
        start_time = datetime.now()
        scroll_count = 0
        
        while (datetime.now() - start_time).total_seconds() < duration:
            if self.behavior.should_scroll(self.profile):
                await self.scroll()
                scroll_count += 1
                
                # Pause to read after scrolling
                pause = self.behavior.uniform(0.8, 2.5)
                await asyncio.sleep(pause)
                
                # Simulate eye movement while reading
                if scroll_count % 3 == 0:
                    await self._simulate_eye_movement()
            else:
                await asyncio.sleep(self.behavior.uniform(0.5, 1.5))
        
        self._log("scroll_session", {"duration": duration, "scroll_count": scroll_count})
    
    # ==================== CLICKING ====================
    
    async def click(
        self,
        selector: str,
        probability: float = None,
        hover_first: bool = True
    ) -> bool:
        """Click with realistic mouse movement and hesitation"""
        if probability is None:
            probability = 0.4 if self.behavior.should_click(self.profile) else 0
        
        if random.random() > probability:
            return False
        
        start_time = datetime.now()
        
        # Hesitation before moving
        await self._hesitate("click")
        
        # Move mouse to element
        position = await self._move_mouse_to_element(selector)
        if not position:
            return False
        
        # Hover before clicking
        if hover_first:
            hover_duration = self.behavior.uniform(0.1, 0.4)
            await asyncio.sleep(hover_duration)
            self._log("hover", {"selector": selector, "duration": hover_duration})
        
        # Final hesitation before click
        await asyncio.sleep(self.behavior.uniform(0.05, 0.2))
        
        # Click
        await self.browser.click(selector)
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._log("click", {"selector": selector, "position": position}, duration_ms)
        
        # Screenshot if enabled
        if self.screenshot_on_action:
            screenshot_dir = Path("screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            await self.browser.screenshot(str(screenshot_dir / f"click_{datetime.now().strftime('%H%M%S')}.png"))
        
        return True
    
    async def click_random(
        self,
        selectors: List[str],
        probability: float = 0.3
    ) -> Optional[str]:
        """Click a random element from list"""
        if not selectors:
            return None
        
        # Shuffle for randomness
        random.shuffle(selectors)
        
        for selector in selectors:
            if await self.click(selector, probability):
                return selector
        
        return None
    
    # ==================== TYPING ====================
    
    async def type_text(
        self,
        selector: str,
        text: str,
        make_mistakes: bool = True,
        clear_first: bool = True
    ):
        """Type text with realistic typing speed and mistakes"""
        if not hasattr(self.browser, 'page') or not self.browser.page:
            return {"characters": 0, "mistakes": 0, "duration_ms": 0}
            
        start_time = datetime.now()
        
        # Move to field
        await self._move_mouse_to_element(selector)
        
        # Click to focus
        await self.browser.click(selector)
        await asyncio.sleep(self.behavior.uniform(0.1, 0.3))
        
        # Clear field if requested
        if clear_first:
            # Select all and delete
            await self.browser.page.keyboard.press("Control+A")
            await asyncio.sleep(self.behavior.uniform(0.05, 0.15))
            await self.browser.page.keyboard.press("Delete")
            await asyncio.sleep(self.behavior.uniform(0.1, 0.2))
        
        # Type each character
        typed_chars = []
        mistakes_made = 0
        
        for i, char in enumerate(text):
            # Check for mistakes
            if make_mistakes and self.behavior.make_mistake(self.profile):
                # Make a mistake
                mistake_char = random.choice("qwertyuiopasdfghjklzxcvbnm")
                await self.browser.page.type(selector, mistake_char, delay=0)
                mistakes_made += 1
                
                # Backspace to correct
                await asyncio.sleep(self.behavior.uniform(0.1, 0.3))
                await self.browser.page.keyboard.press("Backspace")
                await asyncio.sleep(self.behavior.uniform(0.05, 0.15))
            
            # Type correct character
            delay = self.behavior.typing_delay(self.profile, char)
            await self.browser.page.type(selector, char, delay=delay)
            typed_chars.append(char)
            
            # Occasional pause mid-sentence
            if i > 0 and i % random.randint(5, 15) == 0:
                await asyncio.sleep(self.behavior.uniform(0.1, 0.4))
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._log("type", {
            "selector": selector,
            "text_length": len(text),
            "typed_length": len(typed_chars),
            "mistakes": mistakes_made,
            "duration_ms": duration_ms
        })
        
        return {
            "characters": len(typed_chars),
            "mistakes": mistakes_made,
            "duration_ms": duration_ms
        }
    
    # ==================== FORM FILLING ====================
    
    async def fill_form(
        self,
        fields: Dict[str, str],
        pause_between_fields: float = None
    ):
        """Fill multiple form fields with human-like pauses"""
        if pause_between_fields is None:
            pause_between_fields = self.behavior.uniform(0.5, 1.5)
        
        results = {}
        
        for selector, value in fields.items():
            # Type the value
            result = await self.type_text(selector, value)
            results[selector] = result
            
            # Pause between fields
            await asyncio.sleep(pause_between_fields)
        
        return results
    
    # ==================== TAB MANAGEMENT ====================
    
    async def new_tab(self, url: Optional[str] = None):
        """Open new tab with human-like behavior"""
        if not hasattr(self.browser, 'context') or not self.browser.context:
            return None
            
        # Open new tab
        new_page = await self.browser.context.new_page()
        self.tab_counter += 1
        
        # Switch to new tab
        pages = self.browser.context.pages
        self.current_tab_id = len(pages) - 1
        self.browser.page = new_page
        
        # Navigate if URL provided
        if url:
            await self.visit(url, reason="new_tab")
        
        self._log("new_tab", {"url": url, "tab_id": self.current_tab_id})
        
        # Manage tab limit
        if len(pages) > self.max_concurrent_tabs:
            # Close oldest tab
            await pages[0].close()
        
        return self.current_tab_id
    
    async def switch_tab(self, index: int):
        """Switch to different tab"""
        if not hasattr(self.browser, 'context') or not self.browser.context:
            return False
            
        pages = self.browser.context.pages
        
        if 0 <= index < len(pages):
            await asyncio.sleep(self.behavior.uniform(0.3, 0.8))  # Thinking time
            self.browser.page = pages[index]
            self.current_tab_id = index
            self.current_url = self.browser.page.url
            
            self._log("switch_tab", {"from": self.current_tab_id, "to": index})
            
            # Brief pause after switching
            await asyncio.sleep(self.behavior.uniform(0.2, 0.5))
            return True
        
        return False
    
    async def close_tab(self, index: Optional[int] = None):
        """Close tab"""
        if not hasattr(self.browser, 'context') or not self.browser.context:
            return False
            
        if index is None:
            index = self.current_tab_id
        
        pages = self.browser.context.pages
        
        if 0 <= index < len(pages):
            await pages[index].close()
            self._log("close_tab", {"tab_id": index})
            
            # Switch to another tab if needed
            if index == self.current_tab_id and len(pages) > 1:
                await self.switch_tab(0)
            
            return True
        
        return False
    
    # ==================== BACK/FORWARD NAVIGATION ====================
    
    async def go_back(self):
        """Go back in history with hesitation"""
        if self.history_index > 0:
            await self._hesitate("go_back")
            
            self.history_index -= 1
            url = self.page_history[self.history_index]
            await self.visit(url, reason="back")
            
            self._log("go_back", {"url": url})
            return True
        return False
    
    async def go_forward(self):
        """Go forward in history"""
        if self.history_index < len(self.page_history) - 1:
            await self._hesitate("go_forward")
            
            self.history_index += 1
            url = self.page_history[self.history_index]
            await self.visit(url, reason="forward")
            
            self._log("go_forward", {"url": url})
            return True
        return False
    
    # ==================== AI-DRIVEN LINK SELECTION ====================
    
    async def _extract_links(self, html: str, max_links: int = 30) -> List[Tuple[str, str, str]]:
        """Extract visible links with context"""
        # Extract links with anchor text and surrounding context
        link_pattern = r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>([^<]+)</a>'
        links = re.findall(link_pattern, html, re.IGNORECASE)
        
        # Also catch nested HTML in anchor text
        nested_pattern = r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>(.*?)</a>'
        nested_links = re.findall(nested_pattern, html, re.IGNORECASE | re.DOTALL)
        
        all_links = []
        seen = set()
        
        for href, text in links + nested_links:
            # Clean text
            text = re.sub(r'<[^>]+>', '', text).strip()
            if href and text and text not in seen and len(text) < 100:
                # Filter internal/navigation links
                if not href.startswith('#') and not href.startswith('javascript:'):
                    all_links.append((href, text[:60], text))
                    seen.add(text)
                if len(all_links) >= max_links:
                    break
        
        return all_links
    
    async def _select_link_ai(self, html: str, goal: str) -> Optional[str]:
        """Use LLM to select most relevant link"""
        if not self._ensure_llm():
            return None
        
        links = await self._extract_links(html)
        if not links:
            return None
        
        # Build prompt for LLM
        links_text = []
        for i, (url, text, full_text) in enumerate(links[:20]):
            links_text.append(f"{i+1}. URL: {url[:80]}\n   Text: {text}")
        
        prompt = f"""You are a human user browsing the web. Your goal: {goal}

Available links on current page:

{chr(10).join(links_text)}

Choose the SINGLE most relevant link to click to achieve your goal.

Consider:
1. Relevance to goal
2. Link text clarity
3. Likelihood of containing useful information

Return ONLY the link number (1-{len(links)}). No explanation.
"""
        
        try:
            # Call LLM
            if hasattr(self._llm, "safe_generate"):
                result = await self._llm.safe_generate(prompt, timeout=10)
                response = result.get("response", "")
            else:
                response = await self._llm.generate(prompt)
            
            # Extract number
            match = re.search(r"\d+", str(response))
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(links):
                    selected_url = links[idx][0]
                    self._log("ai_link_selection", {
                        "goal": goal,
                        "selected": selected_url,
                        "reasoning": response[:100]
                    })
                    return selected_url
                    
        except Exception as e:
            logger.warning(f"AI link selection failed: {e}")
        
        return None
    
    # ==================== AUTONOMOUS BROWSING ====================
    
    async def smart_browse(
        self,
        start_url: str,
        goal: str = "explore",
        max_pages: int = 5,
        max_time_seconds: int = 300
    ):
        """
        Autonomous browsing with AI-guided decisions
        """
        start_time = datetime.now()
        pages_visited = 1
        
        # Visit start URL
        await self.visit(start_url, reason="start")
        
        while pages_visited < max_pages and (datetime.now() - start_time).total_seconds() < max_time_seconds:
            # Read current page
            await self.scroll_and_read(duration=self.behavior.uniform(5, 15))
            
            # Get page content
            html = await self.browser.get_content()
            
            # Select next link
            if self.use_ai_for_links:
                next_url = await self._select_link_ai(html, goal)
            else:
                links = await self._extract_links(html)
                next_url = random.choice(links)[0] if links else None
            
            # Navigate to selected link
            if next_url and next_url not in self.page_history:
                await self.visit(next_url, reason="ai_selected")
                pages_visited += 1
                
                # Random back/forward navigation
                if self.behavior.should_click(0.15):
                    await asyncio.sleep(self.behavior.uniform(1, 3))
                    if random.random() < 0.5:
                        await self.go_back()
                        pages_visited -= 1
            else:
                # Try to go back if no new links
                if not await self.go_back():
                    break
        
        logger.info(f"Smart browse complete: visited {pages_visited} pages")
        return pages_visited
    
    # ==================== PRE-DEFINED SESSIONS ====================
    
    async def run_research_session(
        self,
        query: str,
        max_pages: int = 10
    ):
        """Run a research-oriented browsing session"""
        await self.smart_browse(
            f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}",
            goal=f"Research about {query}",
            max_pages=max_pages
        )
    
    async def run_shopping_session(
        self,
        product: str,
        max_pages: int = 8
    ):
        """Run a shopping browsing session"""
        await self.smart_browse(
            f"https://www.google.com/search?q=buy+{product.replace(' ', '+')}",
            goal=f"Find best deal for {product}",
            max_pages=max_pages
        )
    
    async def run_social_session(
        self,
        platform: str = "twitter",
        max_pages: int = 15
    ):
        """Run a social media browsing session"""
        urls = {
            "twitter": "https://twitter.com",
            "reddit": "https://reddit.com",
            "facebook": "https://facebook.com"
        }
        
        await self.visit(urls.get(platform, "https://news.ycombinator.com"))
        
        # Social behavior: frequent scrolling, random clicks
        for _ in range(max_pages):
            await self.scroll_and_read(duration=self.behavior.uniform(2, 5))
            
            # Click random links
            html = await self.browser.get_content()
            links = await self._extract_links(html)
            
            if links and self.behavior.should_click(0.6):
                random_link = random.choice(links)[0]
                if random_link.startswith('http'):
                    await self.visit(random_link, reason="social_click")
                    
                    # Stay on page briefly
                    await asyncio.sleep(self.behavior.uniform(5, 15))
                    await self.go_back()
    
    # ==================== MISC ACTIONS ====================
    
    async def take_break(self, duration: float = None):
        """Take a human-like break (no activity)"""
        if duration is None:
            duration = self.behavior.exponential(120)  # Average 2 minutes
        
        self._log("break_start", {"duration": duration})
        await asyncio.sleep(min(duration, 300))  # Max 5 minutes
        self._log("break_end", {"duration": duration})
    
    async def look_around(self):
        """Simulate looking around the page (mouse movements without clicking)"""
        if not hasattr(self.browser, 'page') or not self.browser.page:
            return
            
        viewport = await self.browser.page.evaluate("window.visualViewport")
        width = viewport['width']
        height = viewport['height']
        
        # Move mouse to random points
        for _ in range(random.randint(3, 8)):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            await MouseMovementSimulator.execute_mouse_movement(
                self.browser.page,
                (width // 2, height // 2),
                (x, y),
                duration_ms=random.uniform(200, 500)
            )
            await asyncio.sleep(self.behavior.uniform(0.3, 1.0))
        
        self._log("look_around", {})
    
    async def download_file(self, url: str, save_path: Optional[str] = None):
        """Download file with human-like behavior"""
        if not hasattr(self.browser, 'page') or not self.browser.page:
            return None
            
        await self._hesitate("download")
        
        async with self.browser.page.expect_download() as download_info:
            await self.browser.page.goto(url)
        
        download = await download_info.value
        
        if save_path:
            await download.save_as(save_path)
        else:
            save_path = download.suggested_filename
        
        self.downloads.append({
            "url": url,
            "path": save_path,
            "timestamp": datetime.now().isoformat()
        })
        
        self._log("download", {"url": url, "path": save_path})
        return save_path
    
    # ==================== SCREENSHOTS & VISUAL ====================
    
    async def screenshot_with_heatmap(self, path: str = "screenshot.png"):
        """Take screenshot with eye-tracking heatmap overlay"""
        # Take regular screenshot
        await self.browser.screenshot(path)
        
        # Create heatmap data
        heatmap_data = {
            "fixations": self.eye_fixations[-50:],  # Last 50 fixations
            "trajectories": [t.to_dict() for t in self.trajectories[-20:]]
        }
        
        # Save heatmap data
        heatmap_path = path.replace('.png', '_heatmap.json')
        with open(heatmap_path, 'w') as f:
            json.dump(heatmap_data, f, indent=2)
        
        self._log("screenshot_heatmap", {"path": path, "fixations": len(heatmap_data["fixations"])})
        return path
    
    # ==================== LOGGING ====================
    
    def _log(self, action: str, details: Dict[str, Any], duration_ms: float = None):
        """Log interaction"""
        interaction_time = datetime.now()
        
        if duration_ms is None:
            duration_ms = (interaction_time - self.last_action_time).total_seconds() * 1000
        
        self.interaction_times.append(duration_ms)
        
        self.interactions.append(InteractionRecord(
            action=action,
            details=details,
            timestamp=interaction_time,
            url=self.current_url,
            duration_ms=duration_ms
        ))
        
        self.last_action_time = interaction_time
        
        # Keep interaction history manageable
        if len(self.interactions) > 5000:
            self.interactions = self.interactions[-4000:]
    
    # ==================== UTILITY ====================
    
    async def wait(self, seconds: float):
        """Wait with human-like impatience"""
        self._log("wait_start", {"seconds": seconds})
        
        elapsed = 0
        while elapsed < seconds:
            chunk = min(self.behavior.uniform(0.5, 2.0), seconds - elapsed)
            await asyncio.sleep(chunk)
            elapsed += chunk
            
            # Occasional small movement while waiting
            if random.random() < 0.2:
                await self.look_around()
        
        self._log("wait_end", {"seconds": seconds})
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        avg_interaction_time = sum(self.interaction_times) / max(len(self.interaction_times), 1)
        
        return {
            "session": {
                "duration_sec": round(session_duration, 2),
                "pages_visited": self.pages_visited,
                "total_interactions": len(self.interactions),
                "unique_urls": len(set(self.page_history)),
                "avg_interaction_time_ms": round(avg_interaction_time, 2)
            },
            "behavior": {
                "profile": self.profile.value,
                "eye_movement": self.eye_movement.value,
                "ai_link_selection": self.use_ai_for_links,
                "use_llm": self._llm is not None
            },
            "technical": {
                "trajectories_recorded": len(self.trajectories),
                "eye_fixations": len(self.eye_fixations),
                "downloads": len(self.downloads),
                "session_persistence": self.session_persistence,
                "tab_count": len(self.browser.context.pages) if hasattr(self.browser, 'context') and self.browser.context else 0
            },
            "history": {
                "history_depth": len(self.page_history),
                "current_position": self.history_index,
                "current_url": self.current_url
            }
        }
    
    async def export_session(self, filepath: str):
        """Export session data to JSON"""
        session_data = {
            "metadata": {
                "start_time": self.session_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "profile": self.profile.value,
                "total_pages": self.pages_visited
            },
            "history": self.page_history,
            "interactions": [
                {
                    "action": i.action,
                    "details": i.details,
                    "timestamp": i.timestamp.isoformat(),
                    "url": i.url,
                    "duration_ms": i.duration_ms
                }
                for i in self.interactions
            ],
            "downloads": self.downloads,
            "statistics": await self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session exported to {filepath}")
    
    async def screenshot(self, path: str = "screenshot.png"):
        """Take a screenshot"""
        await self.browser.screenshot(path)
        return path


# ==================== WRAPPER FOR EDIATH ====================

class HumanBrowserWrapper:
    """Wrapper class for EDIATH integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.browser = HumanBrowser(
            headless=config.get("headless", False),
            user_data_dir=config.get("user_data_dir", "data/human_browser_profile"),
            session_persistence=config.get("session_persistence", True),
            use_ai_for_links=config.get("use_ai_for_links", True),
            profile=UserProfile(config.get("profile", "casual")),
            eye_movement=EyeMovement(config.get("eye_movement", "reading")),
            record_trajectories=config.get("record_trajectories", True),
            max_concurrent_tabs=config.get("max_concurrent_tabs", 5),
            screenshot_on_action=config.get("screenshot_on_action", False)
        )
        self.agent_type = "human_browser"
        self.capabilities = [
            "visit", "click", "type", "scroll", "smart_browse", "go_back",
            "go_forward", "new_tab", "switch_tab", "fill_form", "wait",
            "take_break", "run_research_session", "run_shopping_session",
            "get_stats", "export_session", "screenshot"
        ]
    
    async def start(self):
        """Start the browser"""
        await self.browser.start()
    
    async def stop(self):
        """Stop the browser"""
        await self.browser.stop()
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request"""
        action = request.get("action")
        
        if action == "visit":
            success = await self.browser.visit(
                request.get("url", ""),
                reason=request.get("reason", "navigation")
            )
            return {"success": success}
        
        elif action == "click":
            clicked = await self.browser.click(
                request.get("selector", ""),
                probability=request.get("probability")
            )
            return {"success": clicked, "clicked": clicked}
        
        elif action == "type":
            result = await self.browser.type_text(
                request.get("selector", ""),
                request.get("text", ""),
                make_mistakes=request.get("make_mistakes", True)
            )
            return {"success": True, "result": result}
        
        elif action == "scroll":
            await self.browser.scroll(
                request.get("direction", "down"),
                request.get("amount")
            )
            return {"success": True}
        
        elif action == "smart_browse":
            pages = await self.browser.smart_browse(
                request.get("start_url", ""),
                request.get("goal", "explore"),
                max_pages=request.get("max_pages", 5),
                max_time_seconds=request.get("max_time_seconds", 300)
            )
            return {"success": True, "pages_visited": pages}
        
        elif action == "research":
            await self.browser.run_research_session(
                request.get("query", ""),
                max_pages=request.get("max_pages", 10)
            )
            return {"success": True}
        
        elif action == "shopping":
            await self.browser.run_shopping_session(
                request.get("product", ""),
                max_pages=request.get("max_pages", 8)
            )
            return {"success": True}
        
        elif action == "go_back":
            success = await self.browser.go_back()
            return {"success": success}
        
        elif action == "go_forward":
            success = await self.browser.go_forward()
            return {"success": success}
        
        elif action == "new_tab":
            tab_id = await self.browser.new_tab(request.get("url"))
            return {"success": True, "tab_id": tab_id}
        
        elif action == "switch_tab":
            success = await self.browser.switch_tab(request.get("index", 0))
            return {"success": success}
        
        elif action == "fill_form":
            results = await self.browser.fill_form(
                request.get("fields", {}),
                pause_between_fields=request.get("pause_between_fields")
            )
            return {"success": True, "results": results}
        
        elif action == "wait":
            await self.browser.wait(request.get("seconds", 1.0))
            return {"success": True}
        
        elif action == "take_break":
            await self.browser.take_break(request.get("duration"))
            return {"success": True}
        
        elif action == "get_stats":
            stats = await self.browser.get_stats()
            return {"success": True, "stats": stats}
        
        elif action == "export_session":
            await self.browser.export_session(request.get("filepath", "session_export.json"))
            return {"success": True}
        
        elif action == "screenshot":
            path = await self.browser.screenshot(request.get("path", "screenshot.png"))
            return {"success": True, "path": path}
        
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
