from dotenv import load_dotenv
from agents import Agent , AsyncOpenAI , OpenAIChatCompletionsModel, Runner , function_tool , input_guardrail , RunContextWrapper , GuardrailFunctionOutput
import os
from pydantic import BaseModel

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Gemini not defined")

provider = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client=provider
)

class Bank_Details(BaseModel):
    account_name : str
    account_number : str
    pin : int
    
class GuardrailOutput(BaseModel):
    isnot_bank_related : bool


input_guardrail_agent = Agent(
    name = "Input Guardrail agent",
    instructions= "Check If the users query is bank related",
    output_type=GuardrailOutput,
    model=model
)

@input_guardrail
async def check_bank_related(ctx: RunContextWrapper, agent: Agent, input:str) -> GuardrailFunctionOutput:
    result = await Runner.run(
        input_guardrail_agent,
        input,
        context=ctx.context
    )
    return GuardrailFunctionOutput(
        output_info= result.final_output,
        tripwire_triggered= result.final_output.isnot_bank_related
    )
    
def check_user_auth(ctx: RunContextWrapper[Bank_Details], agent: Agent)-> bool:
    if ctx.context.account_name == "Soban Saud" and ctx.context.account_number == "48202873291_@kjdj" and ctx.context.pin == 183093:
        return True
    else:
        return False
    
    
@function_tool(is_enabled=check_user_auth)
def bank_balance(account_number : str) -> str:
    return f"The bank balance of this account_number is 1000000"


user_data = Bank_Details(
    account_name="Soban Saud",
    account_number="48202873291_@kjdj",
    pin= 183093
)

def dynamic_instruction(ctx:RunContextWrapper[Bank_Details],agent:Agent):
    return f"user name is {ctx.context.account_name}  check the users name if its correct use the balance check tool to check thier balance"

bank_agent = Agent(
    name = "Bank Agent",
    instructions= dynamic_instruction,
    model=model,
    tools=[bank_balance],
    input_guardrails= [check_bank_related]
)

result = Runner.run_sync(
    bank_agent,
    input = "What is balance of my account 48202873291_@kjdj ? ",
    context=user_data
)

print("Final Agent Called :- ", result.final_output)