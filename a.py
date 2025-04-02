import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.job import get_current_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("virtual-doorman")
load_dotenv()

# Shared data structure between agents
@dataclass
class VisitData:
    resident_name: Optional[str] = None
    apartment: Optional[str] = None
    visitor_name: Optional[str] = None
    visit_reason: Optional[str] = None

# Mock database functions - replace with real DB calls
async def verify_resident(name: str, apartment: str) -> bool:
    # In real implementation, query your database
    logger.info(f"Checking resident: {name} in apartment {apartment}")
    return True  # Mock validation

async def verify_visitor(resident_name: str, visitor_name: str) -> bool:
    # In real implementation, query your database
    logger.info(f"Checking visitor {visitor_name} for resident {resident_name}")
    return True  # Mock validation

async def log_visit(visit_data: VisitData) -> None:
    logger.info(f"Logging visit: {visit_data}")

class ResidentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a virtual doorman. Your task is to: "
            "1. Greet the resident politely "
            "2. Ask for their full name and apartment number "
            "3. Verify this information matches our records "
            "Keep responses brief and professional.",
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova"),
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def verify_resident_info(
        self, 
        context: RunContext[VisitData],
        resident_name: str,
        apartment: str
    ):
        """Verify resident information against database records."""
        
        is_valid = await verify_resident(resident_name, apartment)
        if not is_valid:
            return None, "Sorry, we couldn't verify your information. Please try again."
        
        # Update shared data
        context.userdata.resident_name = resident_name
        context.userdata.apartment = apartment
        
        # Hand off to VisitorAgent
        visitor_agent = VisitorAgent()
        return visitor_agent, "Thank you! Now let's register your visitor."

class VisitorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a visitor registration system. Your task is to: "
            "1. Ask for the visitor's full name "
            "2. Ask for the reason of the visit "
            "3. Verify this information with building records "
            "Keep responses professional and concise.",
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova"),
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def verify_visitor_info(
        self,
        context: RunContext[VisitData],
        visitor_name: str,
        visit_reason: str
    ):
        """Finalize visitor registration and log the visit."""
        
        # Validate visitor information
        is_valid = await verify_visitor(
            context.userdata.resident_name,
            visitor_name
        )
        
        if not is_valid:
            return None, "Visitor not found in our records. Please check the name and try again."
        
        # Update shared data
        context.userdata.visitor_name = visitor_name
        context.userdata.visit_reason = visit_reason
        
        # Log the visit
        await log_visit(context.userdata)
        
        # Generate final response
        job_ctx = get_current_job_context()
        await job_ctx.room.disconnect()
        return None, "Visitor registered successfully. The resident has been notified. Thank you!"

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession[VisitData](
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=openai.TTS(voice="nova"),
        userdata=VisitData(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=ResidentAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
