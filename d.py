import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

import yaml
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("virtual-doorman")
logger.setLevel(logging.INFO)

voices = {
    "main": "794f9389-aac1-45b6-b726-9d9369183238",
    "visitor": "156fb8d2-335b-4950-9cb3-a2d33befec77",
    "delivery": "6f84f4b8-58a2-430c-8c79-688dad597532",
    "maintenance": "39b376fc-488e-4d0c-8b37-e00b72059fdd",
    "rental": "2a8d5c7f-489b-4f66-9f23-6d901a1c0e54",
}

@dataclass
class UserData:
    resident_name: Optional[str] = None
    apartment_number: Optional[str] = None
    visitor_name: Optional[str] = None
    visit_reason: Optional[str] = None
    maintenance_needs: Optional[str] = None
    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        data = {
            "resident": {
                "name": self.resident_name or "unknown",
                "apartment": self.apartment_number or "unknown",
            },
            "visitor": {
                "name": self.visitor_name or "unknown",
                "reason": self.visit_reason or "unknown",
            },
            "maintenance": self.maintenance_needs or "none",
        }
        return yaml.dump(data)

RunContext_T = RunContext[UserData]

class BaseAgent(Agent):
    async def on_enter(self) -> None:
        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()
        
        if userdata.prev_agent and not isinstance(self.llm, llm.RealtimeModel):
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        chat_ctx.add_message(
            role="system",
            content=f"You are {self.__class__.__name__}. Current data: {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent
        return next_agent, f"Transferring to {name.replace('_', ' ')}"

    def _truncate_chat_ctx(self, items: list[llm.ChatItem], keep_last_n_messages: int = 6, 
                          keep_system_message: bool = False, keep_function_call: bool = False):
        # Same implementation as original restaurant example
        ...

class MainAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the main virtual doorman. Determine if the visitor needs: "
                "1. Resident visit\n2. Delivery\n3. Maintenance\n4. Rental inquiry\n"
                "Use tools to transfer to the appropriate agent."
            ),
            llm=openai.LLM(model="gpt-4o"),
            tts=cartesia.TTS(voice=voices["main"])
        )

    @function_tool()
    async def transfer_visitor(self, context: RunContext_T) -> tuple[Agent, str]:
        """Transfer to visitor agent when someone is visiting a resident"""
        return await self._transfer_to_agent("visitor", context)

    @function_tool()
    async def transfer_delivery(self, context: RunContext_T) -> tuple[Agent, str]:
        """Transfer to delivery agent for package deliveries"""
        return await self._transfer_to_agent("delivery", context)

    @function_tool()
    async def transfer_maintenance(self, context: RunContext_T) -> tuple[Agent, str]:
        """Transfer to maintenance agent for repair requests"""
        return await self._transfer_to_agent("maintenance", context)

    @function_tool()
    async def transfer_rental(self, context: RunContext_T) -> tuple[Agent, str]:
        """Transfer to rental agent for apartment availability"""
        return await self._transfer_to_agent("rental", context)

class VisitorAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Collect resident name and apartment. Validate and notify resident.",
            tools=[self.verify_resident, self.transfer_main],
            tts=cartesia.TTS(voice=voices["visitor"])
        )

    @function_tool()
    async def collect_info(
        self,
        name: Annotated[str, Field(description="Resident's full name")],
        apartment: Annotated[str, Field(description="Apartment number")],
        context: RunContext_T
    ) -> str:
        """Record resident information"""
        userdata = context.userdata
        userdata.resident_name = name
        userdata.apartment_number = apartment
        return f"Recorded {name} in apartment {apartment}"

    @function_tool()
    async def verify_resident(context: RunContext_T) -> str:
        """Verify resident in database and send SMS"""
        userdata = context.userdata
        # Mock DB check
        if userdata.resident_name and userdata.apartment_number:
            return "SMS sent to resident. Please wait for approval."
        return "Resident not found. Please check information."

    @function_tool()
    async def transfer_main(self, context: RunContext_T) -> tuple[Agent, str]:
        """Return to main agent"""
        return await self._transfer_to_agent("main", context)

class DeliveryAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Verify delivery recipient and grant access",
            tools=[self.verify_delivery, self.transfer_main],
            tts=cartesia.TTS(voice=voices["delivery"])
        )

    @function_tool()
    async def verify_delivery(context: RunContext_T) -> str:
        """Check recipient and open door"""
        userdata = context.userdata
        if userdata.resident_name and userdata.apartment_number:
            return "Door opened. Please leave package in lobby."
        return "Recipient not found. Please check information."

class MaintenanceAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Collect maintenance details and log request",
            tools=[self.log_request, self.transfer_main],
            tts=cartesia.TTS(voice=voices["maintenance"])
        )

    @function_tool()
    async def log_request(
        self,
        description: Annotated[str, Field(description="Issue description")],
        context: RunContext_T
    ) -> str:
        """Record maintenance request"""
        userdata = context.userdata
        userdata.maintenance_needs = description
        return "Request logged. Maintenance team will respond shortly."

class RentalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Provide rental info and notify owner",
            tools=[self.list_vacancies, self.transfer_main],
            tts=cartesia.TTS(voice=voices["rental"])
        )

    @function_tool()
    async def list_vacancies(context: RunContext_T) -> str:
        """Show available apartments"""
        return "Available: A101, B202. Owner has been notified."

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = UserData()
    userdata.agents.update({
        "main": MainAgent(),
        "visitor": VisitorAgent(),
        "delivery": DeliveryAgent(),
        "maintenance": MaintenanceAgent(),
        "rental": RentalAgent(),
    })

    agent = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await agent.start(
        agent=userdata.agents["main"],
        room=ctx.room,
        room_input_options={}
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
