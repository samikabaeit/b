import logging
from typing import Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero, turn_detector

logger = logging.getLogger("virtual-doorman")

load_dotenv()


class VirtualDoorman(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a virtual doorman. First ask for the visit purpose (visitor, delivery, maintenance, rental). "
            "Collect information step by step. Always ask one question at a time. "
            "Be polite but concise. Confirm details before taking actions.",
        )
        self.current_purpose: Optional[str] = None
        self.resident_name: Optional[str] = None
        self.apartment_number: Optional[str] = None
        self.visitor_name: Optional[str] = None
        self.reason: Optional[str] = None
        self.maintenance_issue: Optional[str] = None

    async def on_enter(self):
        self.session.generate_reply(instructions="Ask for the visit purpose (visitor/delivery/maintenance/rental)")

    def _check_resident_db(self, name: str, apartment: str) -> bool:
        # Replace with actual DB check
        logger.info(f"Checking DB for {name} in {apartment}")
        return True  # Simulated validation

    def _send_sms(self, number: str, message: str) -> None:
        # Replace with actual SMS integration
        logger.info(f"SMS to {number}: {message}")

    def _get_vacant_units(self) -> list[str]:
        # Replace with actual DB query
        return ["A101", "B202", "C303"]

    @function_tool
    async def handle_purpose(self, context: RunContext, purpose: str) -> dict:
        """Determine visit purpose. Allowed values: visitor, delivery, maintenance, rental"""
        if purpose not in ["visitor", "delivery", "maintenance", "rental"]:
            return {"message": "Invalid purpose. Please choose from visitor/delivery/maintenance/rental"}
        
        self.current_purpose = purpose
        self._reset_session_data()
        
        if purpose == "visitor":
            return {"message": "Visiting a resident. Please provide resident's FULL NAME"}
        elif purpose == "delivery":
            return {"message": "Making a delivery. Please provide resident's FULL NAME"}
        elif purpose == "maintenance":
            return {"message": "Reporting maintenance. Please describe the issue"}
        elif purpose == "rental":
            return {"message": "Rental inquiry. Please wait while I check vacancies..."}

    @function_tool
    async def collect_resident_info(self, context: RunContext, name: Optional[str] = None, 
                                  apartment: Optional[str] = None) -> dict:
        """Collect and validate resident information"""
        if not name:
            return {"message": "Please provide resident's FULL NAME"}
        
        if not apartment:
            self.resident_name = name
            return {"message": "Please provide apartment NUMBER"}
        
        if not self._check_resident_db(name, apartment):
            self._reset_resident_data()
            return {"message": "Resident not found. Please check name and apartment number"}
        
        self.resident_name = name
        self.apartment_number = apartment
        
        if self.current_purpose == "visitor":
            return {"message": "Resident verified. Please provide VISITOR NAME and REASON for visit"}
        elif self.current_purpose == "delivery":
            self._open_door()
            return {"message": "Delivery approved. Door opened. Have a nice day!"}

    @function_tool
    async def handle_visitor_details(self, context: RunContext, visitor_name: Optional[str] = None,
                                   reason: Optional[str] = None) -> dict:
        """Collect visitor details and notify resident"""
        if not visitor_name:
            return {"message": "Please provide VISITOR NAME"}
        
        if not reason:
            self.visitor_name = visitor_name
            return {"message": "Please provide REASON for visit"}
        
        self._send_sms("RESIDENT_PHONE", 
                      f"Visitor: {visitor_name}\nReason: {reason}\nAt: {self.resident_name} {self.apartment_number}")
        return {"message": "Resident notified. They'll be with you shortly. Thank you!"}

    @function_tool
    async def handle_maintenance(self, context: RunContext, issue: Optional[str] = None) -> dict:
        """Record maintenance issue and confirm"""
        if not issue:
            return {"message": "Please describe the MAINTENANCE ISSUE"}
        
        # Replace with actual DB integration
        logger.info(f"Maintenance logged for {self.resident_name}: {issue}")
        return {"message": "Maintenance request recorded. Technician will contact resident shortly"}

    @function_tool
    async def handle_rental(self, context: RunContext, request: Optional[str] = None) -> dict:
        """Provide rental info and notify owner"""
        vacancies = self._get_vacant_units()
        self._send_sms("OWNER_PHONE", "New rental inquiry received")
        return {"message": f"Current vacancies: {', '.join(vacancies)}. Owner has been notified. Thank you!"}

    def _open_door(self) -> None:
        # Replace with actual door control integration
        logger.info("Door opened")

    def _reset_session_data(self) -> None:
        self.resident_name = None
        self.apartment_number = None
        self.visitor_name = None
        self.reason = None
        self.maintenance_issue = None

    def _reset_resident_data(self) -> None:
        self.resident_name = None
        self.apartment_number = None


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4"),
        stt=deepgram.STT(model="nova-3"),
        tts=openai.TTS(voice="ash"),
        turn_detection=turn_detector.EOUModel(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)
    await ctx.wait_for_participant()

    await session.start(
        agent=VirtualDoorman(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
