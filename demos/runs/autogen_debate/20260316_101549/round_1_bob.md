In reducing LLM inference costs, one of the most effective strategies is prompt caching. This technique allows for the storage of previously generated responses, which can be reused for identical or similar prompts. According to the LLM Cost Optimization guide, prompt caching can lead to significant cost savings, potentially reducing inference costs by over 70% (CalmOps, 2023). This approach not only minimizes the number of expensive model invocations but also maintains the quality of responses since cached outputs can be reused without degradation in performance.

Additionally, request batching is another powerful technique. By grouping multiple requests together for simultaneous processing, we can take advantage of the model's parallelism, thereby increasing throughput and reducing overall latency (Clarifai Guide, 2023). 

While model distillation and quantization can also reduce costs, they risk sacrificing model quality, particularly in complex tasks. In contrast, prompt caching and batching maintain the integrity of the outputs while still achieving substantial cost reductions. Thus, focusing on these infrastructure optimizations offers a balanced approach to cost-effectiveness without compromising model performance. 

Sources:
- CalmOps. (2023). LLM Cost Optimization: Reducing Inference Costs 70%+.
- Clarifai Guide. (2023). LLM Inference Optimization Techniques.