### Why structured outputs
- Structured outputs return predictable JSON-like data instead of free text, enabling direct ingestion into databases, APIs, and downstream pipelines without brittle parsing.[1]
- They are foundational for agents and tool use, where the model must pass exact parameters to calculators, search functions, or CRUD operations reliably.[1]

### Two model categories
- Native support: models that implement function calling or JSON mode can adhere to schemas directly via LangChain’s structured-output APIs, giving cleaner, safer results.[1]
- No native support: models lacking these features require output parsers to coerce text into the schema, with validation as a safeguard in application code.[1]

### LangChain pattern: with_structured_output
- Use model.with_structured_output(schema, ...) to request outputs conforming to a given schema; choose method="function_calling" (GPT-style) or JSON mode as supported by the provider.[1]
- Returns either a Python dict or a Pydantic object, depending on how the schema is defined and chosen method.[1]

### Schema options at a glance
- TypedDict: lightweight Python typing; good for quick shaping and IDE help; lacks runtime validation, so treat as guidance for the model.[1]
- Pydantic: robust validation, defaults, Optional handling, coercion, and Field constraints; recommended for Python-centric production use.[1]
- JSON Schema: language-agnostic specification for cross-service contracts and front-end/back-end interoperability; returns dicts in Python.[1]

### Design tips that improve reliability
- Add explicit field descriptions and examples to reduce ambiguity; constrain categoricals using enums or Literal for consistent analytics.[1]
- Mark Optional fields and explicitly instruct the model not to infer or hallucinate missing data; apply validation right after inference to catch issues early.[1]
- Keep schemas minimal and stable; simpler shapes adhere better across providers and reduce agent loop fragility.[1]

### When to choose what
- Prefer Pydantic for Python production pipelines, thanks to validation and coercion; it minimizes runtime errors and data-quality drift.[1]
- Choose JSON Schema for cross-language teams or where schemas must be shared in contracts and tooling (e.g., OpenAPI, AJV).[1]
- Use TypedDict for quick prototypes or low-risk internal scripts where runtime validation isn’t critical.[1]

### Code: TypedDict schema with LangChain
```python
from typing import TypedDict, List, Optional, Literal
from langchain_openai import ChatOpenAI  # or your provider

class ReviewDict(TypedDict, total=False):
    summary: str  # Brief, 1–2 sentences
    sentiment: Literal["POS", "NEG", "NEU"]  # Constrained classes
    key_themes: List[str]
    pros: Optional[List[str]]
    cons: Optional[List[str]]
    reviewer_name: Optional[str]

model = ChatOpenAI(model="gpt-4o-mini")  # example
structured = model.with_structured_output(ReviewDict, method="function_calling")
prompt = "Summarize the review: 'Battery lasts two days, camera is average. Great price.'"
result = structured.invoke(prompt)
# result is a dict, e.g. {"summary": "...", "sentiment": "POS", "key_themes": ["battery","price"], ...}
```


### Code: Pydantic schema with validation
```python
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import List, Optional, Literal
from langchain_openai import ChatOpenAI

class ReviewModel(BaseModel):
    summary: str = Field(description="1–2 sentence summary")
    sentiment: Literal["POS", "NEG", "NEU"] = Field(description="Overall sentiment class")
    key_themes: List[str] = Field(default_factory=list, description="Key topics mentioned")
    pros: Optional[List[str]] = Field(default=None)
    cons: Optional[List[str]] = Field(default=None)
    reviewer_name: Optional[str] = Field(default=None)
    reviewer_email: Optional[EmailStr] = Field(default=None)

    @field_validator("key_themes")
    @classmethod
    def non_empty_themes(cls, v):
        return [t for t in v if t.strip()]  # normalize empties

model = ChatOpenAI(model="gpt-4o-mini")
structured = model.with_structured_output(ReviewModel, method="function_calling")
result_obj = structured.invoke("Review: 'Amazing display, laggy in games, fair price.'")
# Validated Pydantic object
as_dict = result_obj.model_dump()
```


### Code: JSON Schema (language-agnostic)
```python
from langchain_openai import ChatOpenAI

review_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "1–2 sentences"},
        "sentiment": {"type": "string", "enum": ["POS", "NEG", "NEU"]},
        "key_themes": {"type": "array", "items": {"type": "string"}},
        "pros": {"type": ["array","null"], "items": {"type": "string"}},
        "cons": {"type": ["array","null"], "items": {"type": "string"}},
        "reviewer_name": {"type": ["string","null"]},
    },
    "required": ["summary", "sentiment", "key_themes"],
    "additionalProperties": False,
}

model = ChatOpenAI(model="gpt-4o-mini")
structured = model.with_structured_output(
    schema=review_schema, method="json_mode"  # or method="function_calling" if supported
)
data = structured.invoke("Short review: 'Solid speakers, mediocre battery, excellent build.'")
```


### Prompting for Optional fields without hallucination
```python
instruction = """
Extract only fields present in the text.
- Do not infer or guess missing values.
- Use null for missing Optional fields.
- Use sentiment from {POS, NEG, NEU} only.
"""
user_text = "The phone is sturdy and fast; camera is disappointing."
result = structured.invoke(f"{instruction}\n\nText: {user_text}")
```


### Validating and repairing after inference
```python
from pydantic import ValidationError

def validate_or_repair(raw: dict) -> ReviewModel:
    try:
        return ReviewModel(**raw)
    except ValidationError:
        # Simple repair: clamp sentiment, drop unknown keys, ensure arrays
        raw["sentiment"] = raw.get("sentiment") if raw.get("sentiment") in {"POS","NEG","NEU"} else "NEU"
        raw = {k: raw.get(k) for k in {"summary","sentiment","key_themes","pros","cons","reviewer_name"}}
        if not isinstance(raw.get("key_themes"), list): raw["key_themes"] = []
        return ReviewModel(**raw)
```


### Agent/tool use: mapping schema to tools
```python
from typing import TypedDict

class WeatherArgs(TypedDict):
    city: str
    unit: Literal["C","F"]

def get_weather(city: str, unit: str) -> dict:
    # call your weather API here
    return {"city": city, "unit": unit, "temp": 27}

tool_schema = WeatherArgs  # minimal, stable contract

llm = ChatOpenAI(model="gpt-4o-mini")
tool_driver = llm.with_structured_output(tool_schema, method="function_calling")
args = tool_driver.invoke("What's the temperature in Berlin in C?")
weather = get_weather(**args)
```


### Data quality and analytics hints
- Normalize categoricals via enums/Literal to keep BI dashboards consistent and reduce downstream cleaning work.[1]
- Prefer validating immediately after model output and before persistence or tool invocation to avoid propagating bad data into systems.[1]
- Keep output schemas versioned; introduce new fields behind defaults and avoid breaking changes to consuming services.[1]

### Quick glossary
- Function calling: provider feature that returns arguments matching a declared tool/function signature, ideal for agent actions.[1]
- JSON mode: provider directive to emit pure JSON, avoiding extra tokens that break parsers.[1]
- Output parser: code that interprets raw text into a target schema when native structured-output features are unavailable.[1]

### Recap
- Use structured outputs to make LLMs interoperable with systems; in LangChain, prefer with_structured_output with Pydantic for robust Python pipelines, JSON Schema for cross-language contracts, and TypedDict for lightweight prototyping; apply strong field descriptions, enums, Optional discipline, and immediate validation to ensure reliable, production-grade integrations.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/45417937/a27242cf-8977-4c49-8701-4b2331e6494a/paste.txt)
