import json
import ollama
from pydantic import BaseModel

# 1. Create a BaseModel called GeneralResponse
class GeneralResponse(BaseModel):
    answer: str
    topic_category: str

def main():
    models = ['smollm2:135m', 'qwen2.5:0.5b', 'gemma3:270m']

    while True:
        prompt = input("What is your question? (type exit to quit): ")
        if prompt.strip().lower() in ('exit', 'quit'):
            break

        for model_name in models:
            print(f"\nSending prompt to {model_name}: '{prompt}'...")
            
            max_retries = 3
            success = False
            response = None
            content = ""

            for attempt in range(max_retries):
                try:
                    # 2 & 3. Send prompt to the model and force output in JSON format matching the schema
                    response = ollama.chat(
                        model=model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        # Passing the JSON schema to the format argument enforces the output structure
                        format=GeneralResponse.model_json_schema(),
                    )
                    
                    # 4. Parse the response using the GeneralResponse Pydantic model
                    content = response['message']['content']
                    parsed_response = GeneralResponse.model_validate_json(content)
                    
                    print(f"\n=== {model_name} Validated JSON Data ===")
                    print(parsed_response.model_dump_json(indent=2))
                    
                    success = True
                    break  # Validation passed, exit retry loop
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1} Error parsing JSON for {model_name}: {e}")
                    print("Raw content:")
                    print(content)
                    print("-" * 40 + "\n")
            
            if not success or response is None:
                print(f"Failed to get a valid response from {model_name} after {max_retries} attempts.")
                continue

            # 5. Extract metrics
            eval_count = response.get('eval_count', 0)
            eval_duration = response.get('eval_duration', 0) # Time taken in nanoseconds
            prompt_eval_duration = response.get('prompt_eval_duration', 0)
            
            # 6. Calculate the TTFT (Time To First Token) and TPS (Tokens Per Second)
            ttft = prompt_eval_duration / 1e9 if prompt_eval_duration > 0 else 0.0
            
            if eval_duration > 0:
                tps = eval_count / (eval_duration / 1e9)
            else:
                tps = 0.0

            print(f"\n=== {model_name} Performance Metrics ===")
            print(f"Time To First Token (TTFT) : {ttft:.4f} seconds")
            print(f"Tokens Per Second (TPS)    : {tps:.2f} tokens/second")
            print("-" * 40 + "\n")

if __name__ == "__main__":
    main()
