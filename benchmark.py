import json
import ollama
from pydantic import BaseModel

# 1. Create a BaseModel called GeneralResponse
class GeneralResponse(BaseModel):
    answer: str
    topic_category: str

def main():
    while True:
        prompt = input("What is your question for smollm2? (type exit to quit): ")
        if prompt.strip().lower() in ('exit', 'quit'):
            break

        print(f"Sending prompt to smollm2:135m: '{prompt}'...")
        
        # 2 & 3. Send prompt to the model and force output in JSON format matching the schema
        response = ollama.chat(
            model='smollm2:135m',
            messages=[{'role': 'user', 'content': prompt}],
            # Passing the JSON schema to the format argument enforces the output structure
            format=GeneralResponse.model_json_schema(),
        )
        
        # 4. Parse the response using the GeneralResponse Pydantic model
        content = response['message']['content']
        
        try:
            parsed_response = GeneralResponse.model_validate_json(content)
            print("\n=== Validated JSON Data ===")
            print(parsed_response.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print("Raw content:")
            print(content)
            print("-" * 40 + "\n")
            continue

        # 5. Extract metrics
        eval_count = response.get('eval_count', 0)
        eval_duration = response.get('eval_duration', 0) # Time taken in nanoseconds
        prompt_eval_duration = response.get('prompt_eval_duration', 0)
        
        # 6. Calculate the TTFT (Time To First Token) and TPS (Tokens Per Second)
        ttft = prompt_eval_duration / 1e9
        
        if eval_duration > 0:
            tps = eval_count / (eval_duration / 1e9)
        else:
            tps = 0.0

        print("\n=== Performance Metrics ===")
        print(f"Time To First Token (TTFT) : {ttft:.4f} seconds")
        print(f"Tokens Per Second (TPS)    : {tps:.2f} tokens/second")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()
