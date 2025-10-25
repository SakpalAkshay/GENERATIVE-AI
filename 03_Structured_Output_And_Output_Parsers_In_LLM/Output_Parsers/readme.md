


### Why structured outputs
- LLM responses are usually unstructured text; other systems (databases, APIs, tools) need well-formed data, so structured outputs bridge LLMs to those systems via schema-shaped results.[1]
- Models with native structured-output (function calling/JSON mode) are ideal, but parsers let any model produce structured outputs by parsing text into target schemas.[1]

### Two model categories
- Can return structured output: many hosted models with function calling or JSON mode; integrate directly or still optionally use parsers.[1]
- Cannot return structured output: many open-source models; rely on output parsers to impose structure on free text reliably.[1]

### Key parsers compared

| Parser | What it does | When to use | Limitations |
|---|---|---|---|
| StringOutputParser | Extracts the text content from model responses, hiding metadata; clean handoff between steps in a chain | Multi-step chains where only the text is needed between steps | No schema or validation [1] |
| JSONOutputParser | Coaxes JSON and parses it into Python dict | Fastest way to get JSON from a model | Cannot enforce a schema shape; the model decides structure unless very carefully prompted [1] |
| StructuredOutputParser | Enforces a field schema (names/descriptions) and extracts JSON matching it | When exact key structure is required (e.g., fact_1, fact_2, fact_3) | No data-type validation; only schema shape is enforced [1] |
| PydanticOutputParser | Uses a Pydantic model to enforce both schema and data validation | Production pipelines that require types, constraints, and coercion | Requires defining Pydantic models; slightly more setup [1] |

### Translation snippets
- Opening context: हाय गाइ माय नेम इज नितेश एंड यू वेलकम टू माय → “Hi guys, my name is Nitesh and you’re welcome to my channel.”[1]
- Core idea: स्ट्रक्चर्ड आउटपुट में आप अपने एलएलएम को फोर्स करते हो कि वह टेक्चुअल आउटपुट देने के बदले स्ट्रक्चर्ड आउटपुट दे → “In structured output you instruct the LLM to return structured output instead of raw text.”[1]
- On parsers: आउटपुट पार्सर्स लैंगचेन में लिखी हुई क्लासेस हैं जो रॉ टेक्स्ट को स्ट्रक्चर्ड फॉर्मेट में बदलती हैं → “Output parsers are classes in LangChain that convert raw text into structured formats.”[1]

### StringOutputParser
- Purpose: simplify chaining by extracting only the string content, avoiding manual result.content handling; useful for multi-step prompts like “write report” then “summarize”.[1]
- Pattern: Template → Model → StringOutputParser → Template2 → Model → StringOutputParser in a single chain for a clean pipeline.[1]

```python
# String output in a chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
p1 = PromptTemplate.from_template("Write a detailed report on the topic: {topic}")
p2 = PromptTemplate.from_template("Write a five-line summary of the following text:\n{text}")

parser = StrOutputParser()

chain = p1 | model | parser | p2 | model | parser
result = chain.invoke({"topic": "black holes"})
print(result)
```


### JSONOutputParser
- Purpose: request and parse JSON output quickly; add parser.get_format_instructions() into the prompt via partial variables to guide the model to emit JSON.[1]
- Limitation: cannot enforce exact schema keys/shape; the model may choose structure unless strongly steered by prompt.[1]

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

parser = JsonOutputParser()
template = PromptTemplate(
    template=(
        "Give me the name, age, and city of a fictional person.\n{format_instructions}"
    ),
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI(model="gpt-4o-mini")
prompt = template.format()
msg = model.invoke(prompt)
data = parser.parse(msg.content)  # dict with keys decided by model if not constrained
print(type(data), data)
```


### StructuredOutputParser
- Purpose: enforce predefined field names via ResponseSchema descriptors; ideal when downstream expects exact keys like fact_1, fact_2, fact_3.[1]
- Still no data-type validation; shape-only enforcement.[1]

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

schemas = [
    ResponseSchema(name="fact_1", description="Fact one about the topic"),
    ResponseSchema(name="fact_2", description="Fact two about the topic"),
    ResponseSchema(name="fact_3", description="Fact three about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

template = PromptTemplate(
    template="Give three facts about the topic: {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(model="gpt-4o-mini")
msg = model.invoke(template.format(topic="black holes"))
data = parser.parse(msg.content)  # {'fact_1': '...', 'fact_2': '...', 'fact_3': '...'}
print(data)
```


### PydanticOutputParser
- Purpose: enforce schema and validate data types and constraints; ideal for production data extraction, with features like ge/le, regex, EmailStr, Optional, and coercion.[1]
- Flow: define BaseModel → create PydanticOutputParser → include parser.get_format_instructions() in prompt → parse → validated object/dict.[1]

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(ge=18, description="Age of the person (must be >= 18)")
    city: str = Field(description="City the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions()

template = PromptTemplate(
    template=(
        "Generate the name, age, and city of a fictional {place} person.\n{format_instructions}"
    ),
    input_variables=["place"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(model="gpt-4o-mini")
msg = model.invoke(template.format(place="Indian"))
person = parser.parse(msg.content)      # Person instance with validation
print(person.model_dump())              # dict; raises if age < 18 or not int
```


### Choosing the right parser
- Use StringOutputParser for text-to-text multi-step chains where only the content matters between steps.[1]
- Use JSONOutputParser for quick JSON without strict schema needs; fastest to implement but shape may vary.[1]
- Use StructuredOutputParser when specific keys/shape must be enforced but you don’t need type validation.[1]
- Use PydanticOutputParser for strict schema + validation (recommended default for production ingestion).[1]

### Prompting tips
- Insert format instructions from the parser so the LLM knows exact output expectations; use partial_variables in PromptTemplate for clean prompts.[1]
- Keep schemas minimal and stable; simpler shapes adhere better across models and reduce breakage in chains and agents.[1]
- If fields are optional, instruct “do not infer missing values; use null” to reduce hallucinations, then validate/repair post-parse if needed.[1]

### Chain patterns and reliability
- Compose pipelines as chains: Template → Model → Parser; this automatically handles result.content and parse calls, making code shorter and less error-prone.[1]
- If a provider API is unreliable, switch provider (e.g., from a free Hugging Face endpoint to ChatOpenAI) without changing the parser/chain structure; the pattern remains the same.[1]

### Additional useful parsers to explore
- CSV output, list output, markdown, numbered list, enum output, datetime parser, and output-fixing parsers for repairing near-miss responses automatically.[1]
- Documentation lists many more specialized parsers; the same format_instructions pattern applies broadly across them.[1]

### Validation and repair pattern
```python
from pydantic import ValidationError

def validate_or_repair(raw: dict) -> Person:
    try:
        return Person(**raw)
    except ValidationError:
        # Minimal repair strategy
        fixed = {
            "name": str(raw.get("name") or "").strip() or "Unknown",
            "city": str(raw.get("city") or "").strip() or "Unknown",
        }
        # Coerce/guard age
        age = raw.get("age")
        try:
            age = int(age)
        except Exception:
            age = 18
        fixed["age"] = max(18, age)
        return Person(**fixed)
```


### One-paragraph recap
- Use output parsers to convert LLM text into structured, reliable data for systems integration: String for clean text handoff, JSON for quick dicts, Structured for enforcing key shape, and Pydantic for full schema and data validation; wire them with PromptTemplate partial format instructions and chains for concise, robust pipelines that work across models, including those without native structured-output features.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/45417937/157bfc3d-b3a5-4fc0-a9da-b45d46daef0c/paste.txt)
