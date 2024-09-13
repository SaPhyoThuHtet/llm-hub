# llm-hub
Resources, and Practical Codes (using llm, fine-tuning llm, and training model from scratch)

1. **What is a Language Model?** A language model is like having a super-smart language expert in your computer. It learns from lots of text and can predict what words to use in sentences. It's handy for answering questions, translating languages, and generating text. 
A language model is a mathematical model that uses probability and statistics to understand and generate human language. It learns from a dataset of text and can predict the next word or phrase based on context. The prediction of the next word based on the previous words and Language Model could be described as P(Wi∣W0 ​, W1, W2 ​ ,…, Wi−1, LM). The language Model is widely used in various language-related tasks, like translation and answering questions, by calculating probabilities. 

2. **What is a Large Langue Model?** A large language model is a highly advanced AI model with an immense number of parameters and trained on vast datasets. It can understand and generate human language, perform various language-related tasks, and excel at contextual understanding. Despite their versatility, large language models also present challenges related to computational requirements, bias, and ethical concerns.

3. **What is In-Context Learning? (Prompting, Zero-Shot)** In-context learning, also known as prompting, is a method to instruct language models by providing specific input. While effective in various applications, it has limitations. For example, in text generation and translation, the models struggle with longer content due to their limited "context window.
The "context window" is like a frame through which a language model looks at text. It can only see a limited amount of words at a time, typically a few to a few hundred. For example, with a 512-token context window, it can only see the words in the middle and not those too far ahead or behind. This limitation can make it harder for the model to understand and respond well to long or complex text.
In real-time conversations, they may lose context as the conversation progresses. For coding assistance and summarization, complexity or longer texts can lead to less accurate responses. The context window restricts their ability to maintain a complete view of input, affecting coherence. Adapting responses to evolving context is also challenging.

4. **What is Few Shot Learning in LLM?** Few-shot learning in Large Language Models (LLMs) allows these models to generalize from a small number of examples, making them versatile. For instance, LLMs can classify text, translate languages, or answer questions with just a few examples. However, few-shot learning may be less effective with smaller models, complex tasks, and resource limitations due to its need for a large context window. While it offers generalization, its accuracy and robustness may not match traditional supervised learning with larger datasets.

5. **What is Fine Tuning in LLM?** Fine-tuning in Large Language Models (LLMs) is the process of customizing pre-trained language models for specific natural language tasks. It involves training these models on task-specific datasets to make them more proficient in tasks like translation, sentiment analysis, or chatbot responses. Fine-tuning builds on the models' foundational language understanding, allowing them to generate contextually relevant and accurate text for specialized applications. This approach saves time and resources compared to training models from scratch and is widely used in various natural language processing tasks.
Fine-tuning a language model involves adjusting the model's parameters (θ) to minimize a task-specific loss function (J) using an optimization algorithm. The goal is to find the optimal parameters (θ*) that make the model perform well on the specific task. This process uses the gradient (∇J(θ)) and a learning rate (η) to guide parameter updates, enhancing the model's performance for that particular task.

6. **What is Insturction Fine Tuning?** Instruction fine-tuning in Large Language Models (LLMs) is a training technique that involves modifying a pre-trained language model by providing specific instructions or examples during a fine-tuning process. This fine-tuning helps the model generate more precise and controlled outputs based on the provided instructions. It allows the model to better understand and follow explicit directions in its responses.
Model Parameters: Let θ represent the model's parameters.
Loss Function with Instructions: Define a loss function J(θ, I) that considers both the model parameters (θ) and the provided instructions (I). This loss function guides the model to adhere to the instructions: J(θ, I) = L(y, f(x, θ, I)) Where: L(y, f(x, θ, I)) is the loss for a given example, where y is the target output, x is the input, θ is the model parameters, and I represents the instructions.
Optimization with Instructions: Use an optimization algorithm to minimize the loss with respect to the model parameters while considering the provided instructions: θ* = argmin J(θ, I)

7. **What is Full Fine Tuning?** Full fine-tuning in Large Language Models (LLMs) involves training the model extensively from scratch on a specific task or domain. This process adjusts all of the model's parameters based on a custom dataset, making it highly specialized for the target application, such as translation or chatbot responses. It enables the LLM to exhibit task-specific behavior, but it demands significant computational resources and access to the model's training infrastructure.
Mathematically, it can be described as follow.
Model Parameters:
Let θ represent the model's parameters, including weights (W) and biases (b). 
Loss Function:
Define a loss function (J) that quantifies the model's error in performing a specific task. The loss is typically defined as the sum of losses over all examples in the fine-tuning dataset: J(θ) = Σ L(y_i, f(x_i, θ)) Where: J(θ) is the overall loss. L(y_i, f(x_i, θ)) is the loss for each example i, which measures the error between the predicted output (f(x_i, θ)) and the true output (y_i). Σ represents the summation over all examples in the fine-tuning dataset.
Fine-Tuning Objective:
The goal of full fine-tuning is to find the optimal model parameters (θ*) that minimize the loss: θ* = argmin J(θ)
Optimization Algorithm:
To achieve this, you typically use an optimization algorithm, such as stochastic gradient descent (SGD). This algorithm iteratively updates the model's parameters in the direction that minimizes the loss: θ=θ - η * ∇J(θ) Where: η is the learning rate, which controls the size of each parameter update. ∇J(θ) is the gradient of the loss with respect to the model parameters, indicating the direction in which the parameters should be adjusted to reduce the loss
In summary, full fine-tuning mathematically involves iteratively adjusting the model's parameters (θ) using an optimization algorithm (e.g., SGD) to minimize a loss function (J), which measures the error in performing a specific task. This process allows the model to adapt and specialize for the defined task or domain.

**8. Quantization**: Quantization is a process employed to transform the numerical representation of model weights, usually stored as 32-bit floating-point values, into lower-precision formats like 16-bit float, 16-bit int, 8-bit int, or even 4/3/2-bit int. This approach brings about various benefits, including a reduction in model size, expedited fine-tuning, and swifter inference times. Particularly in resource-constrained settings such as single-GPU setups or mobile edge devices, where computing resources are limited, quantization becomes a necessity for efficient model fine-tuning and accelerated inference.

**9. Model Hallucination**: Large language model hallucination refers to a phenomenon where a sophisticated language model, like GPT-3, generates text that appears to be coherent and contextually relevant but is actually fictional or inaccurate. In other words, the model creates information that is not based on real data or facts. This can happen when the model tries to generate text in response to a prompt but lacks accurate or specific information about the topic. As a result, it may generate text that sounds plausible but is essentially made up by the model. Hallucination is a significant concern when it comes to the reliability and trustworthiness of information generated by such models, and it highlights the importance of careful fact-checking and verification when using their outputs.

**10. What is Retrieval-Augmented Generation?**: Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to improving LLM output so it remains relevant, accurate, and useful in various contexts.

**11. What is RLHF?:** Reinforcement Learning with Human Feedback (RLHF) is a specialized approach in machine learning, especially relevant in cases where traditional reinforcement learning faces challenges. It integrates human guidance into the training process of AI agents. Instead of relying solely on environmental rewards, RLHF allows humans to provide feedback to the agent, helping it understand what actions are desirable. This feedback is used to create reward models that enhance the agent's learning process. The agent is trained using a combination of these reward models and traditional environmental rewards, optimizing its behaviour. RLHF is valuable in complex tasks like robotics and autonomous driving, ensuring AI systems align better with human values and safety. It offers a powerful method for refining AI behaviour in scenarios where defining a precise reward function is challenging.
In mathematical terms, this can be expressed as: 
![image](https://github.com/user-attachments/assets/41e3fae2-69dd-4d56-bd46-e3c3f8341f1a)


However, RL faces challenges when it comes to defining suitable reward functions, especially in complex, high-dimensional environments where human expertise is valuable. This is where RLHF comes in. In RLHF, human feedback is introduced to enhance the learning process. Human feedback can be represented as a function:
![image](https://github.com/user-attachments/assets/bcf6254b-9904-4169-97ad-2d31b13d0bf0)


This human-provided reward signal (rh) is then integrated with the traditional environmental rewards, creating a combined reward signal that guides the agent's learning process. This can be represented mathematically as:
![image](https://github.com/user-attachments/assets/b7a7600c-337e-4f9e-aa62-fc16bdb537e6)

### 12. What is Catastrophic Forgetting in Fine Tuning?
What is Catastrophic Forgetting in Fine-Tuning?
Catastrophic forgetting in fine-tuning occurs when a neural network, pre-trained on one task and then fine-tuned for a new task, loses the ability to perform well on the original task. The fine-tuning process can lead to significant changes in the model's representation, causing it to forget knowledge relevant to the pre-training task. Mitigating this issue is crucial, and strategies like regularization, learning rate scheduling, larger datasets, and auxiliary tasks can help maintain the model's performance on both tasks.

Let represent θ as weights and L as Loss.
Original Task Loss: L_original(θ_initial)
Fine-Tuning Task Loss: L_new(θ_fine-tuned)
Total Loss during Fine-Tuning with Weighted Combination: Total Loss = α * L_original(θ_fine-tuned) + (1 - α) * L_new(θ_fine-tuned) 
Optimization during Fine-Tuning to Find θ_fine-tuned: 
θ_fine-tuned = argmin(α * L_original(θ_fine-tuned) + (1 - α) * L_new(θ_fine-tuned)) 

The issue of catastrophic forgetting arises from the fact that optimizing for the new task (L_new) during fine-tuning can lead to changes in the model's parameters (θ_fine-tuned) that negatively affect the original task's performance (L_original). The balance between these two tasks is controlled by the weighting factor α.


### LLAMA2
Fine Tuning llama2:
1. https://artificialcorner.com/mastering-llama-2-a-comprehensive-guide-to-fine-tuning-in-google-colab-bedfcc692b7f
2. https://medium.com/ai-in-plain-english/fine-tuning-llama2-0-with-qloras-single-gpu-magic-1b6a6679d436
3. https://anirbansen2709.medium.com/finetuning-llms-using-lora-77fb02cbbc48

### BERT
Fine Tuning BERT
1. https://towardsdatascience.com/fine-tune-a-large-language-model-with-python-b1c09dbc58b2



## Notebooks

## Resources:
### Github:
Cheat Sheet of LLM: https://github.com/Abonia1/CheatSheet-LLM

### Blogs/Articles with practical code:
1. General: How to train a language model from scratch: https://huggingface.co/blog/how-to-train
2. llama2: https://huggingface.co/blog/llama2#how-to-prompt-llama-2 (If you are using Google Colab, you would need to upgrade to the pro version)
3. Information Extraction with llama2: https://khadkechetan.medium.com/information-extraction-with-llm-chetan-kkhadke-cc41674b380
4. LLAMA2 with LangChain: https://medium.com/@mayuresh.gawai/implementation-of-llama-v2-in-python-using-langchain-%EF%B8%8F-ebebe82e881b
5. Getting Started With LLama2: https://ai.meta.com/llama/get-started/
6. Text Embedding with GPT4all: https://docs.gpt4all.io/gpt4all_python_embedding.html
 
### Courses:
1. General: Generative AI with LLMS (Coursera): https://www.coursera.org/learn/generative-ai-with-llms
   This is a great course and you can acquire the basics of LLMs, Fine Tuning with FLAN-T5 Model, and Ethical Usage.
2. General: CS324 - Large Language Models, https://stanford-cs324.github.io/winter2022/
   Theories from this course are worth learning.
3. ChatGPT: ChatGPT Prompt Engineering for Developers

### LLM for Under-resourced Languages:

### RAG
https://jayant017.medium.com/rag-using-langchain-part-2-text-splitters-and-embeddings-3225af44e7ea
https://www.youtube.com/watch?v=tcqEUSNCn8I
### Ethical LLM
1. carbon emissions estimations on the hub: https://huggingface.co/blog/carbon-emissions-on-the-hub
2. Codecarbon to measure carbon emissions: https://codecarbon.io/
3. Mlco2 to measure carbon emission: https://mlco2.github.io/impact/

### References:
Coming Soon
   


