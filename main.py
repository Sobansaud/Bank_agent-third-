from dotenv import load_dotenv
from agents import (
    Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner,
    function_tool, input_guardrail, RunContextWrapper, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered, output_guardrail, OutputGuardrailTripwireTriggered
)
import os
import asyncio
from pydantic import BaseModel

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Gemini not defined")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# ---------------- MODELS ---------------- #
class Message(BaseModel):
    output: str

class Bank_Details(BaseModel):
    account_name: str
    account_number: str
    pin: int

class GuardrailOutput(BaseModel):
    isnot_bank_related: bool


# ---------------- GUARDRAIL AGENTS ---------------- #
input_guardrail_agent = Agent(
    name="Input Guardrail agent",
    instructions="Check If the users query is bank related",
    output_type=GuardrailOutput,
    model=model
)

output_guardrail_agent = Agent(
    name="Output Guardrail agent",
    instructions="Check if the output includes any bank related information",
    output_type=GuardrailOutput,
    model=model
)


@input_guardrail
async def check_bank_related(ctx: RunContextWrapper, agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(
        input_guardrail_agent,
        input,
        context=ctx.context
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.isnot_bank_related
    )


@output_guardrail
async def check_output_bank_related(ctx: RunContextWrapper, agent: Agent, output: Message) -> GuardrailFunctionOutput:
    result = await Runner.run(
        output_guardrail_agent,
        output,
        context=ctx.context
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.isnot_bank_related
    )


# ---------------- AUTH & TOOL ---------------- #
def check_user_auth(ctx: RunContextWrapper[Bank_Details], agent: Agent) -> bool:
    return (
        ctx.context.account_name == "Soban Saud"
        and ctx.context.account_number == "48202873291_@kjdj"
        and ctx.context.pin == 183093
    )


@function_tool(is_enabled=check_user_auth)
def bank_balance(account_number: str) -> str:
    return f"The bank balance of this {account_number} is 1000000"


# ---------------- CONTEXT ---------------- #
user_data = Bank_Details(
    account_name="Soban Saud",
    account_number="48202873291_@kjdj", #  WRONG ACCOUNT BCZ CHECK SUPPORT AGENT IS WORK OR NOT
    pin=183093
)




def dynamic_instruction(ctx: RunContextWrapper[Bank_Details], agent: Agent):
    return f"user name is {ctx.context.account_name}. If correct, use the balance check tool to check their balance."


# ---------------- HANDOFF AGENTS ---------------- #
general_agent = Agent(
    name="General Q&A Agent",
    instructions="Answer general questions that are not bank related.",
    model=model
)

support_agent = Agent(
    name="Support Agent",
    instructions="Handle failed authentication attempts. Politely tell the user that their authentication failed and guide them to support.",
    model=model
)


# ---------------- MAIN BANK AGENT ---------------- #
bank_agent = Agent(
    name="Bank Agent",
    instructions=dynamic_instruction,
    model=model,
    tools=[bank_balance],
    input_guardrails=[check_bank_related],
    output_guardrails=[check_output_bank_related],
)


# ---------------- DRIVER FUNCTION ---------------- #
async def main():
    user_query = input("Enter your query: ")

    try:
        # Run main bank agent
        result = await Runner.run(
            bank_agent,
            user_query,
            context=user_data
        )
        print("✅ Guardrail didn't tripwire:", result.final_output)
        
    except Exception as e:
        # If auth fails, send to Support Agent
        print("❌ Authentication Failed → handoff to Support Agent")
        result = await Runner.run(support_agent, user_query)
        print("Support Agent Response:", result.final_output)

    except InputGuardrailTripwireTriggered:
        # handoff to general agent
        print("⚠️ Input Guardrail tripped → handoff to General Agent")
        result = await Runner.run(general_agent, user_query)
        print("General Agent Response:", result.final_output)

    except OutputGuardrailTripwireTriggered:
        # handoff to general agent
        print("⚠️ Output Guardrail tripped → handoff to General Agent")
        result = await Runner.run(general_agent, user_query)
        print("General Agent Response:", result.final_output)




# ---------------- RUN ---------------- #
asyncio.run(main())
