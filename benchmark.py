import json
import ollama
from pydantic import BaseModel, ValidationError

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

        model_results = []

        for model_name in models:
            print(f"\nSending prompt to {model_name}: '{prompt}'...")
            
            max_retries = 3
            success = False
            response = None
            content = ""
            retries_needed = 0

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
                    retries_needed = attempt
                    break  # Validation passed, exit retry loop
                    
                except ValidationError as e:
                    print(f"   ⚠️ JSON format failed. Retrying attempt {attempt + 1}/3...")
                    print(f"Error parsing JSON for {model_name}: {e}")
                    print("Raw content:")
                    print(content)
                    print("-" * 40 + "\n")
                except Exception as e:
                    print(f"   ⚠️ Unexpected error. Retrying attempt {attempt + 1}/3...")
                    print(f"Error for {model_name}: {e}")
                    print("Raw content:")
                    print(content)
                    print("-" * 40 + "\n")
            
            if not success or response is None:
                print(f"❌ Model failed to output valid JSON after {max_retries} tries")
                model_results.append({
                    'name': model_name,
                    'retries': max_retries,
                    'tps': 0.0,
                    'success': False
                })
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

            model_results.append({
                'name': model_name,
                'retries': retries_needed,
                'tps': tps,
                'success': True
            })

        # --- AUTOMATED JUDGE ---
        print("\n" + "=" * 40)
        print("🏆 Final Analysis & Recommendation")
        print("=" * 40)

        successful_models = [res for res in model_results if res['success']]

        if not successful_models:
            print("No models were able to successfully generate a valid JSON response.\n")
            continue

        # 1. Reliability first: models with 0 retries
        perfect_models = [res for res in successful_models if res['retries'] == 0]

        best_model = None
        if perfect_models:
            # 2. Highest TPS among 0-retry models
            best_model = max(perfect_models, key=lambda x: x['tps'])
        else:
            # 3. Fewest retries, then Highest TPS
            best_model = sorted(successful_models, key=lambda x: (x['retries'], -x['tps']))[0]

        other_summaries = []
        for res in model_results:
            if res['name'] != best_model['name']:
                if res['success']:
                    if res['retries'] == 0:
                        other_summaries.append(f"{res['name']} also needed 0 retries but was slower ({res['tps']:.2f} TPS)")
                    else:
                        other_summaries.append(f"{res['name']} required {res['retries']} retries")
                else:
                    other_summaries.append(f"{res['name']} failed completely")

        comparison_text = ", whereas " + " and ".join(other_summaries) if other_summaries else ""

        if best_model['retries'] == 0:
            print(f"{best_model['name']} is the recommended model here because it followed the JSON schema perfectly on the first try and maintained a solid speed of {best_model['tps']:.2f} TPS{comparison_text}.\n")
        else:
            print(f"{best_model['name']} is the recommended model here because it required the fewest retries ({best_model['retries']}) and achieved a speed of {best_model['tps']:.2f} TPS{comparison_text}.\n")

if __name__ == "__main__":
    main()
