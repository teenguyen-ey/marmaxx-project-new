from openai import AzureOpenAI
 
openai_deployment = "gpt-4o"
 
client = AzureOpenAI(
    api_key="9406341a64f9426b8ecab78b5453ab93",
    api_version="2024-08-01-preview",
    azure_endpoint="https://ue2coai54baoa01.openai.azure.com/"
)
backend_db = "no"  # change to "yes" when you want real references instead of placeholders
 
# Instruction for placeholder-only mode
instruction_placeholder = """
You are a helpful assistant.
"""
 
instruction_real = """
You are a helpful assistant. Use the provided context to answer the query.
You are preparing a formal utility regulatory response.
"""
 
Instruction = instruction_placeholder if backend_db == "no" else instruction_real
 
context = """
Q1: Does Georgia Power Company currently measure resilience quantitatively or qualitatively? If so, please provide the results of the Company's resilience measurements. If not, does the Company know if its system is resilient?
Answer:- The Company is not aware of any standard industry measures for resilience. Therefore, the Company's resilience considerations are qualitative and focus on its general ability to adapt, respond, and recover during disruptive events. While the Company has not experienced a high-impact low probability man-initiated disaster, the Company has consistently demonstrated resilience during and following natural disasters such as Hurricane Michael, through its ability to respond and minimize the impact of disruption. Georgia Power received the EEI (Edison Electric Institute) Emergency Recovery Award in eight of the last 15 years. It is given to select EEI member companies to recognize their efforts to restore power to customers after service interruptions caused by severe weather events or other natural disasters.
"""
query = "Provide a copy of all accounting procedures or guidelines that describe the accounting for storm costs, including, but not limited to, the definition of storm costs, both expense and capital, including the specific expenses that qualify as “O&M expense” and the definition of “capital costs”; the definition of “incremental costs”; and the circumstances and timing for commencing deferral of carrying costs, if any; and the assignment and/or allocation to functions, i.e., generation, transmission, distribution, other; and the assignment and/or allocation to jurisdictions, to the extent that is applicable."
 
prompt = f"""Context:{context}
Query:{query}
Instruction:{Instruction}"""
 
response = client.chat.completions.create(
    model=openai_deployment,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)
 
print(response.choices[0].message.content)