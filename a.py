import logging
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Callable
from pathlib import Path
from functools import wraps
from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    Agent, AgentSession, ChatContext, JobContext, JobProcess,
    RoomInputOptions, RoomOutputOptions, RunContext, WorkerOptions,
    cli, metrics, function_tool
)
from livekit.plugins import deepgram, openai, silero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            'doorman.log', maxBytes=10*1024*1024, backupCount=5)
    ]
)
logger = logging.getLogger("smart-doorman")
load_dotenv()

# --------------- Configuration ---------------
CONVERSATION_LOG_DIR = Path("./conversations")
CONVERSATION_LOG_DIR.mkdir(exist_ok=True)

class APIMetrics:
    """Custom metrics collector for external API calls"""
    def __init__(self):
        self._counters = metrics.PrometheusCounterCollection()
        self._durations = metrics.PrometheusDurationCollection()
    
    def track_call(self, name: str):
        """Decorator to track API call metrics"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.monotonic()
                try:
                    result = await func(*args, **kwargs)
                    self._counters.add(name, 1, {"status": "success"})
                    return result
                except Exception as e:
                    self._counters.add(name, 1, {"status": "error"})
                    logger.error(f"API call failed: {name}", exc_info=True)
                    raise
                finally:
                    duration = time.monotonic() - start_time
                    self._durations.observe(name, duration)
            return wrapper
        return decorator

api_metrics = APIMetrics()

# --------------- Data Models ---------------
@dataclass
class DoormanData:
    intent: Optional[str] = None
    resident_name: Optional[str] = None
    apartment: Optional[str] = None
    visitor_name: Optional[str] = None
    maintenance_issue: Optional[str] = None
    vacancies: List[str] = None

# --------------- Services with Metrics & Logging ---------------
class DatabaseService:
    @staticmethod
    @api_metrics.track_call("db_query")
    async def find_resident(name: str, apartment: str) -> bool:
        logger.info(f"Checking resident: {name} in {apartment}")
        return True  # Implement actual DB logic

    @staticmethod
    @api_metrics.track_call("db_query")
    async def get_vacancies() -> List[str]:
        logger.info("Fetching vacant apartments")
        return ["501", "302", "105"]

class SMSService:
    @staticmethod
    @api_metrics.track_call("sms_send")
    async def send(number: str, message: str) -> bool:
        logger.info(f"Sending SMS to {number}: {message[:50]}...")
        return True  # Integrate with actual SMS gateway

class DoorService:
    @staticmethod
    @api_metrics.track_call("door_open")
    async def open() -> bool:
        logger.info("Triggering door open mechanism")
        return True  # Implement hardware integration

# --------------- Conversation Logger ---------------
class ConversationLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_file = CONVERSATION_LOG_DIR / f"{session_id}.jsonl"
        self._ensure_header()

    def _ensure_header(self):
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                json.dump({
                    "session_id": self.session_id,
                    "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }, f)
                f.write("\n")

    async def log_interaction(self, role: str, content: str):
        entry = {
            "timestamp": time.time(),
            "role": role,
            "content": content
        }
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")

# --------------- Agents with Error Handling ---------------
def handle_errors(func: Callable):
    """Decorator for error handling and conversation logging"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            session = getattr(self, "session", None)
            if session:
                await session.conversation_logger.log_interaction("agent", func.__name__)
            return await func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Agent error in {self.__class__.__name__}: {e}", exc_info=True)
            if session:
                await session.room.disconnect()
            raise
    return wrapper

class MainAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="Main concierge - route requests",
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova")
        )

    @handle_errors
    async def on_enter(self):
        await self.session.generate_reply()

    @function_tool
    @handle_errors
    async def route_conversation(self, context: RunContext[DoormanData], intent: str):
        """Route to appropriate service agent"""
        context.userdata.intent = intent.lower()
        agent_map = {
            "visitor": VisitorAgent,
            "delivery": DeliveryAgent,
            "maintenance": MaintenanceAgent,
            "rental": RentalAgent
        }
        
        agent_class = agent_map.get(context.userdata.intent)
        if not agent_class:
            return None, "Invalid request type"
        
        return agent_class(), f"Connecting to {agent_class.__name__}"

# (Implement other agents similarly with @handle_errors decorator)

# --------------- System Initialization ---------------
def prewarm(proc: JobProcess):
    """Initialize shared resources"""
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD model loaded successfully")
    except Exception as e:
        logger.error("Failed to initialize VAD", exc_info=True)
        raise

async def entrypoint(ctx: JobContext):
    """Main entry point with session setup"""
    try:
        await ctx.connect()
        session_id = ctx.room.name
        
        session = AgentSession[DoormanData](
            vad=ctx.proc.userdata["vad"],
            llm=openai.LLM(model="gpt-4o-mini"),
            stt=deepgram.STT(model="nova-3"),
            tts=openai.TTS(voice="nova"),
            userdata=DoormanData()
        )
        
        # Initialize conversation logging
        session.conversation_logger = ConversationLogger(session_id)
        
        # Setup metrics collection
        usage_collector = metrics.UsageCollector()
        session.on("metrics_collected", lambda ev: (
            metrics.log_metrics(ev.metrics),
            usage_collector.collect(ev.metrics)
        ))
        
        # Add custom API metrics to prometheus
        metrics.registry.register_many([
            api_metrics._counters,
            api_metrics._durations
        ])
        
        ctx.add_shutdown_callback(lambda: logger.info(
            f"Session {session_id} metrics: {usage_collector.get_summary()}"))
        
        await session.start(
            agent=MainAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                audio_processing={"echo_cancellation": True}
            ),
            room_output_options=RoomOutputOptions(
                transcription_enabled=True
            )
        )
        
    except Exception as e:
        logger.error("Fatal initialization error", exc_info=True)
        raise

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        prometheus_port=9090  # Expose metrics endpoint
    ))
