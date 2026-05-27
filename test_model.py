from langchain_ollama import OllamaLLM

print("Connecting to local MediQuery model...")
llm = OllamaLLM(model="mediquery_local")

# 2. Setup the medical context
mock_context = "Patient is a 45-year-old male presenting with acute hypertension and a documented history of severe asthma. Peer-reviewed literature states that non-selective beta-blockers are strictly contraindicated in asthmatic patients due to the risk of bronchospasm."
mock_question = "Can I prescribe a non-selective beta-blocker to this patient?"

# 3. Format exactly as trained
prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly precise medical research assistant. Answer the user's question using ONLY the provided contexts. Do not invent information. If the contexts contradict, defer to the PEER-REVIEWED LITERATURE.<|eot_id|><|start_header_id|>user<|end_header_id|>
Context data:
{mock_context}

Question: {mock_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# 4. Generate the response
print("\nThinking...")
response = llm.invoke(prompt)

print("\n--- MediQuery Final Output ---")
print(response.strip())
print("\n------------------------------")