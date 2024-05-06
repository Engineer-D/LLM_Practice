# https://medium.com/aimonks/reflection-agents-with-langgraph-agentic-llm-based-applications-87e43c27adc7#:~:text=So%20basically%20to%20achieve%20reflection,to%20make%20the%20content%20better. 
#*******************************Library**************************************
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from decouple import config
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#****************************************************************************

#*********************GENERATE AGENT*****************************************
prompt = ChatPromptTemplate.from_messages(
    [
        #  this needs to be a tuple so do not forget the , at the end and do not include any , in-between
        (
            "system",
            "You are an AI assistant researcher tasked with researching on a variety of topics in a short summary of 5 paragraphs."
            " Generate the best research possible as per user request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(api_key=config("OPENAI_API_KEY"))


generate = prompt | llm


research = ""
request = HumanMessage(
    content="Research on climate change"
)

# test the single generate agent
for chunk in generate.stream({"messages": [request]}):
    print(chunk.content, end="")
    research += chunk.content

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        #  this needs to be a tuple so do not forget the , at the end
        (
            "system",
            "You are a senior researcher"
            " Provide detailed recommendations, including requests for length, depth, style, etc."
            " to an asistant researcher to help improve this researches",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm

print("\n\n\n")
print("============Reflect===============")
print("\n\n\n")

reflection = ""
for chunk in reflect.stream({"messages": [request, 
                                          HumanMessage(content=research)]}):
    print(chunk.content, end="")
    reflection += chunk.content


print("\n\n\n")
print("============REVISED===============")
print("\n\n\n")

revised = ""
# test the single generate agent
for chunk in generate.stream({"messages": [request, 
                                           HumanMessage(content=research), 
                                           AIMessage(content=reflection)]}):
    print(chunk.content, end="")
    revised += chunk.content