# Local SLM Benchmarking: Speed vs. Quality

## Methodology
This project benchmarks local Small Language Models (SLMs) to evaluate the trade-offs between inference speed and output quality. The testing framework is written in **Python** and utilizes the **Ollama** Python library to interface with models running locally. To guarantee consistent and machine-readable outputs, we leverage **Pydantic** to enforce strictly validated JSON schema outputs directly from the language models.

## Results
The primary performance metrics evaluated are Time To First Token (TTFT) and overall generation speed in Tokens Per Second (TPS).

| Model          | Parameters | TTFT (s) | TPS     | Notes / Accuracy |
|----------------|------------|----------|---------|------------------|
| `smollm2:135m` | 135M       | 0.0262   | 160.34  | Severe Hallucinations |

## Conclusion
The initial benchmark reveals a significant trade-off between raw speed and factual recall with edge-level micro-models. While the `smollm2:135m` model achieves massive generation speeds (>150 TPS) and near-instant response times, it suffers from severe factual hallucinations. For instance, when tasked with recalling factual data (e.g., the population of the capital of Japan), it generated completely fabricated numbers (claiming well over 1.9 trillion). 

This demonstrates that ultra-small models in the 100M-300M parameter range are poorly equipped for tasks requiring internal factual knowledge recall. Instead, these blazing-fast models are far better suited for simple logic routing, formatting, or contextual extraction when explicitly provided with the ground-truth text.
