# LangChain Chains: Sequential, Parallel, and Conditional

This guide distills the translated transcript into practical notes and runnable code for building chains in LangChain, including sequential, parallel, and conditional flows, plus tips on LCEL (LangChain Expression Language) composition and Runnables.[10]

## Why chains

- Chains connect steps like prompt construction, model invocation, and parsing so each step’s output becomes the next step’s input; this reduces glue code and makes complex apps maintainable.[10]
- Beyond linear pipelines, chains support parallel branches and conditional routing to design multi-output and decision-driven applications.[10]

## Core building blocks

- PromptTemplate: structure prompts with placeholders via input_variables for reuse and clarity.[10]
- Chat models: interchangeable providers (e.g., OpenAI, Anthropic) can be swapped without changing chain structure.[10]
- Output parsers:
  - StrOutputParser for passing clean text between steps.[10]
  - PydanticOutputParser for deterministic, validated outputs used in routing or persistence.[10]
- LCEL: compose steps with the pipe operator |; visualize pipelines with ASCII graphs for debugging/reviews [10].

## Minimal sequential chain

A simple three-step pipeline: PromptTemplate → Chat model → String parser, invoked with one input and returning final text.[10]

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = PromptTemplate(
    template="Generate five interesting facts about {topic}.",
    input_variables=["topic"],
)
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "cricket"})
print(result)

# Visualize the chain topology (optional)
chain.get_graph().print_ascii()
```


Notes:
- The parser removes metadata and returns just the text, making multi-step chains cleaner.[10]
- The graph output documents intent in PRs and helps reviewers follow the flow.[10]

## Two-step sequential chain (report → summary)

Generate a detailed report first, then summarize it into five bullet points with a second prompt—still one composed chain.[10]

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

report_prompt = PromptTemplate(
    template="Generate a detailed report on {topic}.",
    input_variables=["topic"],
)
summary_prompt = PromptTemplate(
    template="Generate a five-point summary from the following text:\n{text}",
    input_variables=["text"],
)

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = (
    report_prompt
    | model
    | parser
    | summary_prompt
    | model
    | parser
)

result = chain.invoke({"topic": "Unemployment in India"})
print(result)
chain.get_graph().print_ascii()
```


Tip:
- Keep subchains small and composable so you can swap models or parsers without refactoring the flow.[10]

## Parallel chain (notes + quiz, then merge)

Given a long document, produce concise study notes and a five-question quiz in parallel (two branches) and then merge the results into a single study document.[10]

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableParallel

notes_prompt = PromptTemplate(
    template="Generate short and simple study notes from the following text:\n{text}",
    input_variables=["text"],
)
quiz_prompt = PromptTemplate(
    template="Generate five short question-answer pairs from the following text:\n{text}",
    input_variables=["text"],
)
merge_prompt = PromptTemplate(
    template="Merge the provided notes and quiz into a single study document.\nNotes:\n{notes}\n\nQuiz:\n{quiz}",
    input_variables=["notes", "quiz"],
)

model_openai = ChatOpenAI(model="gpt-4o-mini")
model_anthropic = ChatAnthropic(model="claude-3-haiku-20240307")  # example model
parser = StrOutputParser()

notes_chain = notes_prompt | model_openai | parser
quiz_chain = quiz_prompt | model_anthropic | parser

parallel = RunnableParallel({"notes": notes_chain, "quiz": quiz_chain})
merge_chain = merge_prompt | model_openai | parser

chain = parallel | merge_chain

text = "A long explanatory document about linear regression: assumptions, OLS, bias-variance, residual diagnostics, R^2 vs adjusted R^2, regularization, etc."
result = chain.invoke({"text": text})
print(result)
chain.get_graph().print_ascii()
```


Why parallel:
- Reduces latency for independent tasks and keeps responsibilities isolated per branch before merging into a final artifact.[10]

## Conditional chain (route by sentiment)

Classify feedback as positive/negative, then generate an appropriate reply; ensure deterministic routing values by enforcing structured output from the classifier.[10]

```python
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Overall sentiment")

parser_cls = PydanticOutputParser(pydantic_object=Feedback)
cls_prompt = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback into 'positive' or 'negative' only.\n"
        "{format_instructions}\n\nFeedback:\n{feedback}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser_cls.get_format_instructions()},
)

model = ChatOpenAI(model="gpt-4o-mini")
str_parser = StrOutputParser()

classify_chain = cls_prompt | model | parser_cls

pos_prompt = PromptTemplate(
    template="Write a warm, appreciative reply to this positive feedback:\n{feedback}",
    input_variables=["feedback"],
)
neg_prompt = PromptTemplate(
    template="Write an empathetic, helpful reply to this negative feedback:\n{feedback}",
    input_variables=["feedback"],
)

pos_chain = pos_prompt | model | str_parser
neg_chain = neg_prompt | model | str_parser

branch = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", pos_chain),
    (lambda x: x["sentiment"] == "negative", neg_chain),
    RunnableLambda(lambda _: "Could not determine sentiment."),
)

chain = classify_chain | branch

print(chain.invoke({"feedback": "Fantastic customer service and speedy delivery!"}))
print(chain.invoke({"feedback": "Battery dies quickly and support never responded."}))

chain.get_graph().print_ascii()
```


Why structure the classifier output:
- Literal-constrained Pydantic output keeps routing logic stable (no fragile free-form strings), which is critical when branches trigger different actions or integrations.[10]

## Visualization

Document chains in PRs with ASCII graphs to clarify IO flow and aid review without running the code.[10]

```python
chain.get_graph().print_ascii()
```


## Practical tips

- Prefer small, declarative chains using PromptTemplate → Model → Parser blocks; they are easier to test and refactor.[10]
- Use PydanticOutputParser whenever downstream logic depends on exact values or types (routing, database writes, tool calls).[10]
- Swap providers freely; chains remain stable as long as IO contracts (prompt inputs/outputs) are preserved.[10]

## What’s next

- Learn Runnables and LCEL internals to understand how sequential, parallel, and conditional execution is implemented, and how to wrap utility functions via RunnableLambda for fallbacks and glue logic.[10]

[1](https://python.langchain.com/api_reference/langchain/chains.html)
[2](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.base.Chain.html)
[3](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html)
[4](https://www.youtube.com/watch?v=-Ueh5XBpcoY)
[5](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.query_constructor.base.construct_examples.html)
[6](https://docs.langchain.com)
[7](https://www.langchain.com)
[8](https://langchain-doc.readthedocs.io/en/latest/index.html)
[9](https://github.com/langchain-ai/langchain)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/45417937/45443184-9b2d-4b78-afe5-1783fa807edf/paste.txt)
