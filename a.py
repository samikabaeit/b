import logging
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    Agent, AgentSession, ChatContext, JobContext, JobProcess,
    RoomInputOptions, RoomOutputOptions, RunContext, WorkerOptions,
    cli, metrics, function_tool
)
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("smart-doorman")
load_dotenv()

# Shared data structure for all agents
@dataclass
class DoormanData:
    intent: Optional[str] = None
    resident_name: Optional[str] = None
    apartment: Optional[str] = None
    visitor_name: Optional[str] = None
    maintenance_issue: Optional[str] = None
    vacancies: List[str] = None

# Mock database and services - implement these!
async def find_resident(name: str, apartment: str) -> bool:
    """Check resident exists in database"""
    return True  # Implement real DB check

async def get_vacancies() -> List[str]:
    """Get list of available apartments"""
    return ["501", "302", "105"]  # Mock data

async def send_sms(number: str, message: str) -> bool:
    """Send SMS notification"""
    logger.info(f"SMS to {number}: {message}")
    return True

async def open_door() -> bool:
    """Trigger door opening mechanism"""
    logger.info("Door opened")
    return True

# -------------------- Agents -------------------- 
class MainAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are the main concierge. Determine user's need from: "
            "1. Visitor registration 2. Delivery 3. Maintenance 4. Rental info "
            "Ask clarifying questions if needed. Be polite and professional.",
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova")
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def route_conversation(self, context: RunContext[DoormanData], intent: str):
        """Route to appropriate specialist based on determined intent"""
        intent = intent.lower()
        context.userdata.intent = intent
        
        agents = {
            "visitor": VisitorAgent,
            "delivery": DeliveryAgent,
            "maintenance": MaintenanceAgent,
            "rental": RentalAgent
        }
        
        if intent not in agents:
            return None, "Sorry, I didn't understand. Please try again."
        
        return agents[intent](), f"Connecting you to {intent.capitalize()} services"

class VisitorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="Handle visitor registration. Collect: "
            "- Resident full name - Apartment number "
            "- Visitor name - Send SMS notification to resident",
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova")
        )

    @function_tool
    async def register_visitor(self, context: RunContext[DoormanData],
                             resident_name: str, apartment: str, visitor_name: str):
        """Finalize visitor registration"""
        if not await find_resident(resident_name, apartment):
            return None, "Resident not found"
            
        context.userdata.resident_name = resident_name
        context.userdata.apartment = apartment
        context.userdata.visitor_name = visitor_name
        
        # Implement real SMS gateway integration
        await send_sms("+1234567890", 
                      f"Visitor {visitor_name} arrived for {resident_name}")
        
        await context.session.room.disconnect()
        return None, "Resident notified. Visitor registered!"

class DeliveryAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="Handle package deliveries. Verify resident info and open door",
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova")
        )

    @function_tool
    async def handle_delivery(self, context: RunContext[DoormanData],
                            resident_name: str, apartment: str):
        """Process delivery request"""
        if not await find_resident(resident_name, apartment):
            return None, "Resident not found"
        
        await open_door()
        await context.session.room.disconnect()
        return None, "Door opened. Please leave package in lobby."

class MaintenanceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="Handle maintenance requests. Collect: "
            "- Resident name - Apartment - Issue description",
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova")
        )

    @function_tool
    async def log_request(self, context: RunContext[DoormanData],
                         issue: str):
        """Record maintenance issue"""
        context.userdata.maintenance_issue = issue
        # Implement real ticketing system integration
        await send_sms("+1987654321", 
                      f"Maintenance needed {context.userdata.apartment}: {issue}")
        
        await context.session.room.disconnect()
        return None, "Request logged. Technician will arrive within 2 hours."

class RentalAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="Provide rental information. List vacancies and notify owner",
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="nova")
        )

    @function_tool
    async def list_vacancies(self, context: RunContext[DoormanData]):
        """Show available apartments and notify owner"""
        vacancies = await get_vacancies()
        context.userdata.vacancies = vacancies
        
        await send_sms("+1122334455", 
                      "New rental inquiry received - please follow up")
        
        await context.session.room.disconnect()
        return None, f"Available units: {', '.join(vacancies)}. Owner notified."

# -------------------- System Setup --------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession[DoormanData](
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=openai.TTS(voice="nova"),
        userdata=DoormanData()
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()
    session.on("metrics_collected", lambda ev: (
        metrics.log_metrics(ev.metrics),
        usage_collector.collect(ev.metrics)
    ))
    
    ctx.add_shutdown_callback(lambda: logger.info(
        f"Usage Summary: {usage_collector.get_summary()}"))

    await session.start(
        agent=MainAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True)
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
