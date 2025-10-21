
### LangChain Prompt Fundamentals

The material explains prompt fundamentals in LangChain, contrasts static vs dynamic prompts, shows how to build a Streamlit research assistant and a console chatbot, introduces message types and chat histories, and teaches `PromptTemplate`, `ChatPromptTemplate`, and `MessagesPlaceholder` for robust multi-turn applications.

***

### What is a Prompt?

A prompt is the message sent to an LLM. While prompts can be multimodal (image, audio, video), the focus here is on text prompts as used in most applications today. Prompt wording heavily influences the output; small changes can shift responses significantly, which is why prompt design and prompt engineering have become critical practices.

### Temperature Clarified

Temperature is a parameter that controls the randomness and creativity of an LLM's output.
*   **Near 0 (e.g., 0.0 to 0.2):** Yields more repeatable, deterministic outputs for identical inputs. This is ideal for factual, precision-based tasks.
*   **Higher values (e.g., 1.0 to 1.5):** Generate more diverse, creative, and sometimes unexpected outputs for the same input. This is useful for brainstorming or creative writing.

```python
# Example of setting temperature in a LangChain model
from langchain_openai import ChatOpenAI

# Low temperature for predictable output
llm_deterministic = ChatOpenAI(temperature=0.0)

# High temperature for creative output
llm_creative = ChatOpenAI(temperature=1.2)

# Identical inputs will produce nearly identical results with llm_deterministic
# but varied results with llm_creative.
```

### Static vs. Dynamic Prompts

*   **Static Prompts:** Raw strings passed directly to the model. They are simple to implement but are fragile, as users can misspell words, alter constraints, or cause an inconsistent user experience.
*   **Dynamic Prompts:** Use a predefined template with placeholders. User inputs are collected in a constrained manner (e.g., through dropdowns, sliders) to fill the template, ensuring consistency and providing guardrails.

### `PromptTemplate` Essentials

`PromptTemplate` allows developers to define a template string with specified `input_variables`. The template can then be filled using the `.invoke()` method to produce a final prompt. Its main benefits over simple f-strings include:
*   **Validation:** Built-in validation of placeholders (`validate_template=True`).
*   **Reusability:** Templates can be saved to and loaded from JSON format.
*   **Integration:** Natively integrates with LangChain's chain components.

```python
from langchain.prompts import PromptTemplate

# 1. Define the template with placeholders
template_string = "Summarize the key findings of the paper titled '{paper_title}' in a {style} style. The summary should be approximately {length} words."
prompt_template = PromptTemplate.from_template(
    template=template_string
)

# 2. Fill the template with user inputs
final_prompt = prompt_template.invoke({
    "paper_title": "Attention Is All You Need",
    "style": "simple",
    "length": "100"
})

print(final_prompt)
# Expected output is a LangChain `PromptValue` object, which can be passed to a model.
```

### Streamlit Research Assistant

A simple UI can be built with Streamlit to create a research assistant. Key components include:
*   A header (`st.header`) for the title.
*   Controlled inputs like `st.selectbox` for the paper title, writing style (e.g., "simple," "math-heavy"), and summary length.
*   A button (`st.button`) to trigger the summarization.
*   The model's response (`result.content`) is displayed on the page.

Using structured inputs instead of a free-form text box reduces typos and enforces constraints for more reliable outputs.

### LLM Invocation Patterns

*   **Single Message:** Invoking the model with a one-off prompt, which can be either a static string or a dynamically generated prompt from a `PromptTemplate`.
*   **Multi-Message:** Passing a list of messages for a multi-turn conversation. This is crucial for chatbots that need context from previous turns.

### Building a Console Chatbot

A basic console chatbot starts with a `while` loop that reads user input until "exit" is typed. The initial version lacks context awareness. To add chat history:
1.  Initialize a list to store the conversation history.
2.  In each loop, append the user's message and the model's response.
3.  Send the entire history list to the model on each turn so it can reference prior context.

```python
# Simplified console chatbot with history
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

chat_history = [SystemMessage(content="You are a helpful AI assistant.")]
model = ChatOpenAI()

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    chat_history.append(HumanMessage(content=user_input))
    
    result = model.invoke(chat_history)
    ai_response = result.content
    
    chat_history.append(AIMessage(content=ai_response))
    print(f"AI: {ai_response}")
```

### Message Roles and Labeling

LangChain uses three core message types to structure conversations:
*   `SystemMessage`: Sets the overall role, instructions, or persona for the AI (e.g., "You are a helpful assistant specializing in astrophysics.").
*   `HumanMessage`: Represents the input from the user.
*   `AIMessage`: Represents the output from the model.

Properly labeling messages is essential for preventing confusion and improving model behavior, especially in long conversations.

### `ChatPromptTemplate` and `MessagesPlaceholder`

For dynamic multi-turn conversations, use `ChatPromptTemplate`. A special placeholder, `MessagesPlaceholder`, can be used to inject an entire list of messages (like the chat history) at runtime.

This pattern is highly effective for building context-aware chatbots:

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a prompt template that includes a placeholder for chat history
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty comedian who tells jokes about {domain}."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{topic}")
])

# During runtime, invoke with the history and new topic
final_prompt = chat_prompt.invoke({
    "domain": "programming",
    "chat_history": [
        HumanMessage(content="Why did the programmer quit his job?"),
        AIMessage(content="Because he didn't get arrays.")
    ],
    "topic": "Tell me another one."
})
```

### Chains with Prompts and Models

Instead of invoking the prompt and then the model separately, you can chain them together. This allows a single `.invoke()` call to fill the template and pass the result to the model in one seamless step. This is the preferred approach in larger applications.

```python
# Chaining a prompt template with a model
chain = chat_prompt | model

# A single invoke call now handles both steps
response = chain.invoke({
    "domain": "science",
    "chat_history": [],
    "topic": "What's the best thing about Switzerland?"
})

print(response.content)
```

### Quick How-To Checklists

*   **Turn a Static Prompt into a Dynamic One:**
    1.  Define a `PromptTemplate` with placeholders (e.g., `{paper_input}`, `{style_input}`).
    2.  Collect values using a structured UI (like Streamlit `selectboxes`).
    3.  Create the final prompt: `prompt = template.invoke({...})`.
    4.  Invoke the model: `result = model.invoke(prompt)`.

*   **Add Memory to a Chatbot:**
    1.  Initialize `chat_history` with a `SystemMessage`.
    2.  In each turn, append the `HumanMessage(user_input)`.
    3.  Invoke the model with the full `chat_history`.
    4.  Append the returned `AIMessage(result.content)` to the history.
    5.  Use `MessagesPlaceholder` in a `ChatPromptTemplate` to manage history injection cleanly.

*   **Save and Reuse Templates:**
    1.  Save a template: `template.save("my_template.json")`.
    2.  Load it later: `from langchain.prompts import load_prompt; loaded_template = load_prompt("my_template.json")`.

### Common Pitfalls and Fixes

*   **Missing Placeholders or Extra Keys:** Enable `validate_template=True` in your `PromptTemplate` to catch mismatches during development.
*   **`ChatPromptTemplate` Substitution Confusion:** Use the recommended role-message tuple style (`("system", "...")`) or the `from_messages` constructor to ensure variables are filled correctly.
*   **No Context in Follow-ups:** Always maintain and resend the labeled chat history. Without it, the model will not be able to reference prior turns in the conversation.
