import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("user-info-agent")
logger.setLevel(logging.INFO)

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a customer service agent. Collect the user's name, apartment number, and city. "
                "Start by asking for their name. When provided, call get_user_name. "
                "Then ask for their apartment number and call get_apartment_number. "
                "Finally ask for their city and call get_city. "
                "Confirm all details at the end. Be friendly and conversational."
            ),
        )
        self.user_name = None
        self.apartment_number = None
        self.city = None

    @function_tool()
    async def get_user_name(self, name: str) -> str:
        """Called when the user provides their name. Store the name."""
        self.user_name = name
        return f"Got it! Your name is {name}."

    @function_tool()
    async def get_apartment_number(self, apartment_number: str) -> str:
        """Called when the user provides their apartment number. Store the number."""
        self.apartment_number = apartment_number
        return f"Apartment number {apartment_number} recorded."

    @function_tool()
    async def get_city(self, city: str) -> str:
        """Called when the user provides their city. Store the city."""
        self.city = city
        return f"City {city} saved. Thank you!"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
    )

    await ctx.wait_for_participant()
    await agent.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
