In the pursuit of reducing LLM inference costs in production, I argue that prompt-level and routing strategies, particularly shorter prompts, cascading from cheaper to more expensive models, and adaptive model selection, are the most effective and easiest to implement. Shorter prompts reduce the computational load by requiring less processing power and memory, thus leading to cost savings without compromising output quality significantly (Brown et al., 2020). 

Cascading prompts allow for initial queries to be handled by less expensive models, reserving more complex processing for those that genuinely require it. This tiered approach minimizes resource wastage and optimally allocates computational resources (Joulin et al., 2017). Adaptive model selection further enhances efficiency by dynamically choosing the model based on the task’s complexity, ensuring that resources are utilized only when necessary.

While techniques like model distillation and quantization offer substantial benefits, they often require extensive retraining or may lead to quality degradation. In contrast, the aforementioned routing strategies can be implemented with existing frameworks, making them more accessible and practical for immediate application (Vaswani et al., 2017). 

Sources:
- Brown, T. B., et al. (2020). "Language Models are Few-Shot Learners."
- Joulin, A., et al. (2017). "Bag of Tricks for Efficient Text Classification."
- Vaswani, A., et al. (2017). "Attention is All You Need."