In addressing the question of reducing LLM inference costs, I argue that implementing prompt-level and routing strategies, particularly shorter prompts and adaptive model selection, are some of the most effective and simplest methods. Shorter prompts inherently reduce the number of tokens processed, leading to direct cost savings. According to research, LLM providers charge based on both input and output tokens, so minimizing token count can significantly lower expenses (Aron Hack, 2023).

Moreover, routing strategies allow for the dynamic selection of cheaper models based on the complexity of the task. By utilizing cheaper models for simpler queries, organizations can optimize resource allocation and cut costs further without sacrificing performance. This cascading from cheaper to more expensive models ensures that only high-complexity tasks utilize the more costly resources (A16Z, 2023).

While other techniques such as model distillation and quantization are valuable, they often require more extensive infrastructure changes and engineering effort, making them less accessible for immediate implementation. Thus, focusing on prompt optimization and adaptive routing offers a practical and cost-effective approach to managing LLM inference costs. 

Sources:
- Aron Hack. "15 Proven Strategies to Reduce LLM Costs Without Sacrificing Performance." 
- A16Z. "Welcome to LLMflation - LLM inference cost is going down fast."