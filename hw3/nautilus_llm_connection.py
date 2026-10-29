# This program is to demonstrate how to use the OpenAI Python client library 
# to interact with the NRP LLM API for generating chat completions.
from dotenv import load_dotenv
import os
from openai import OpenAI

# To load the environment variables from a .env file that contains key-value pairs. 
load_dotenv()

# You can create an instance of the OpenAI client with the API key and base URL.
# Using OpenAI library does not mean you are connecting to the OpenAI.
# You uses the OpenAI library to connect to the NRP LLM or other LLMs hosted on servers.
client = OpenAI(
    # The key-value pair, OPENAI_API_KEY=your-generated-api-key can be saved in 
    # an environment variable but in this example, it is saved in the .env file 
    # and loaded by calling load_dotenv().
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://ellm.nrp-nautilus.io/v1",
)

response = client.chat.completions.create(
    model="qwen3", # You can change the model to any other model available on the NRP LLM server.
    # You can send multiple messages to the LLM model, each with a role and content. 
    # The roles can be: system, user, assistant, or tool.
    # system is instructions/context given to the AI model
    # user is the model's input from the user
    # assistant is the previous model response to provide the context for the model's next response
    # tool is the result from a tool call
    max_tokens=500,
    messages=[
        {"role": "system", "content": "You are an experienced recruiter at a top technology company. Give practical, honest, and actionable advice to new graduates about improving their chances of being recruited by leading technology companies."},
        {"role": "user", "content": "How can I improve my chances of being recruited by a top technology company as a new graduate? Should I focus on improving my resume, building projects, developing technical skills, networking, or gaining other experience? Please recommend specific steps I can take."
        "respond using paragraphs only, without bullet points or numbered lists, do not use bold text such as using '**text**'. "
        },
    ],
)

# Print the generated response from the model. 
# The response is contained in the 'choices'. Why is choices an array? 
# Because the model may return multiple completions for a single request, allowing you to
# choose from different responses. 
# In this case, we access the first response.
print(response.choices[0].message.content)
