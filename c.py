import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, cartesia, silero

logger = logging.getLogger("virtual-doorman")
logger.setLevel(logging.INFO)

# Mock database - replace with real DB calls
mock_db = {
    "residents": {
        ("John Doe", "A101"): {"phone": "+1234567890", "valid": True},
        ("Jane Smith", "B202"): {"phone": "+0987654321", "valid": True},
    },
    "vacant_apartments": ["C303", "D404"],
    "maintenance_requests": []
}

@dataclass
class UserData:
    resident_name: Optional[str] = None
    apartment_number: Optional[str] = None
    visitor_name: Optional[str] = None
    visit_reason: Optional[str] = None
    maintenance_needs: Optional[str] = None
    current_agent: Optional[str] = None
    agents: dict[str, Agent] = field(default_factory=dict)

    def summarize(self) -> str:
        return f"""
        Resident: {self.resident_name or 'N/A'} ({self.apartment_number or 'N/A'})
        Visitor: {self.visitor_name or 'N/A'} - Reason: {self.visit_reason or 'N/A'}
        Maintenance: {self.maintenance_needs or 'N/A'}
        """

RunContext_T = RunContext[UserData]

class BaseAgent(Agent):
    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        userdata.current_agent = name
        return userdata.agents[name], f"Transferring to {name} agent"

class MainAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="You are the main virtual doorman. Determine if the visitor is here for: "
                        "1. Visiting a resident\n2. Delivery\n3. Maintenance request\n4. Rental inquiry\n"
                        "Use tools to transfer to the appropriate agent.",
            llm=openai.LLM(model="gpt-4o"),
            tts=cartesia.TTS()
        )

    @function_tool()
    async def to_visitor_agent(self, context: RunContext_T) -> tuple[Agent, str]:
        """Transfer to visitor agent when someone is visiting a resident"""
        return await self._transfer_to_agent("visitor", context)

    @function_tool()
    async def to_delivery_agent(self, context: RunContext_T) -> tuple[Agent, str]:
        """Transfer to delivery agent for package deliveries"""
        return await self._transfer_to_agent("delivery", context)

    @function_tool()
    async def to_maintenance_agent(self, context: RunContext_T) -> tuple[Agent, str]:
        """Transfer to maintenance agent for repair requests"""
        return await self._transfer_to_agent("maintenance", context)

    @function_tool()
    async def to_rental_agent(self, context: RunContext_T) -> tuple[Agent, str]:
        """Transfer to rental agent for apartment availability inquiries"""
        return await self._transfer_to_agent("rental", context)

class VisitorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="Collect resident name and apartment number. Validate and notify resident.",
            tools=[self.check_resident],
            tts=cartesia.TTS()
        )

    @function_tool()
    async def collect_resident_info(
        self,
        name: Annotated[str, Field(description="Full name of resident")],
        apartment: Annotated[str, Field(description="Apartment number")],
        context: RunContext_T
    ) -> str:
        """Record resident information"""
        context.userdata.resident_name = name
        context.userdata.apartment_number = apartment
        return f"Recorded resident {name} in apartment {apartment}"

    @function_tool()
    async def check_resident(context: RunContext_T) -> str:
        """Check if resident exists in database and notify them"""
        userdata = context.userdata
        resident = mock_db["residents"].get((userdata.resident_name, userdata.apartment_number))
        
        if resident and resident["valid"]:
            # In real implementation, send SMS here
            return (f"Resident verified. SMS sent to {resident['phone']}. "
                    "Please wait for the resident to approve your visit.")
        return "Resident not found. Please check the information and try again."

class DeliveryAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="Verify delivery recipient and grant access if valid",
            tools=[self.verify_delivery],
            tts=cartesia.TTS()
        )

    @function_tool()
    async def verify_delivery(context: RunContext_T) -> str:
        """Check recipient and open door if valid"""
        userdata = context.userdata
        resident = mock_db["residents"].get((userdata.resident_name, userdata.apartment_number))
        
        if resident and resident["valid"]:
            # In real implementation, trigger door unlock here
            return "Delivery verified. Main door opened. Please leave package in lobby."
        return "Recipient not found. Please check the information."

class MaintenanceAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="Collect maintenance details and schedule repair",
            tools=[self.log_maintenance],
            tts=cartesia.TTS()
        )

    @function_tool()
    async def log_maintenance(
        self,
        description: Annotated[str, Field(description="Description of the issue")],
        context: RunContext_T
    ) -> str:
        """Record maintenance request"""
        context.userdata.maintenance_needs = description
        mock_db["maintenance_requests"].append(description)
        return "Request logged. Maintenance team will contact you shortly."

class RentalAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="Provide rental information and notify property manager",
            tools=[self.list_vacancies],
            tts=cartesia.TTS()
        )

    @function_tool()
    async def list_vacancies(context: RunContext_T) -> str:
        """Show available apartments and notify manager"""
        vacancies = ", ".join(mock_db["vacant_apartments"])
        # In real implementation, send SMS to owner
        return f"Available apartments: {vacancies}. Property manager has been notified."

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = UserData()
    userdata.agents = {
        "main": MainAgent(),
        "visitor": VisitorAgent(),
        "delivery": DeliveryAgent(),
        "maintenance": MaintenanceAgent(),
        "rental": RentalAgent(),
    }

    agent = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(),
        vad=silero.VAD(),
        max_tool_steps=5,
    )

    await agent.start(
        agent=userdata.agents["main"],
        room=ctx.room,
        room_input_options={}
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
