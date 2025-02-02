<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Agents - Guided tour

[[open-in-colab]]

In this guided visit, you will learn how to build an agent, how to run it, and how to customize it to make it work better for your use-case.

### Building your agent

To initialize a minimal agent, you need at least these two arguments:

- `model`, a text-generation model to power your agent - because the agent is different from a simple LLM, it is a system that uses a LLM as its engine. You can use any of these options:
    - [`TransformersModel`] takes a pre-initialized `transformers` pipeline to run inference on your local machine using `transformers`.
    - [`HfApiModel`] leverages a `huggingface_hub.InferenceClient` under the hood and supports all Inference Providers on the Hub.
    - [`LiteLLMModel`] similarly lets you call 100+ different models and providers through [LiteLLM](https://docs.litellm.ai/)!
    - [`AzureOpenAIServerModel`] allows you to use OpenAI models deployed in [Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service).

- `tools`, a list of `Tools` that the agent can use to solve the task. It can be an empty list. You can also add the default toolbox on top of your `tools` list by defining the optional argument `add_base_tools=True`.

Once you have these two arguments, `tools` and `model`,  you can create an agent and run it. You can use any LLM you'd like, either through [Inference Providers](https://huggingface.co/blog/inference-providers), [transformers](https://github.com/huggingface/transformers/), [ollama](https://ollama.com/), [LiteLLM](https://www.litellm.ai/), or [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service).

<hfoptions id="Pick a LLM">
<hfoption id="HF Inference API">

HF Inference API is free to use without a token, but then it will have a rate limit.

To access gated models or rise your rate limits with a PRO account, you need to set the environment variable `HF_TOKEN` or pass `token` variable upon initialization of `HfApiModel`. You can get your token from your [settings page](https://huggingface.co/settings/tokens)

```python
from smolagents import CodeAgent, HfApiModel

model_id = "meta-llama/Llama-3.3-70B-Instruct" 

model = HfApiModel(model_id=model_id, token="<YOUR_HUGGINGFACEHUB_API_TOKEN>") # You can choose to not pass any model_id to HfApiModel to use a default free model
# you can also specify a particular provider e.g. provider="together" or provider="sambanova"
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Local Transformers Model">

```python
# !pip install smolagents[transformers]
from smolagents import CodeAgent, TransformersModel

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = TransformersModel(model_id=model_id)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="OpenAI or Anthropic API">

To use `LiteLLMModel`, you need to set the environment variable `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, or pass `api_key` variable upon initialization.

```python
# !pip install smolagents[litellm]
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key="YOUR_ANTHROPIC_API_KEY") # Could use 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Ollama">

```python
# !pip install smolagents[litellm]
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2", # This model is a bit weak for agentic behaviours though
    api_base="http://localhost:11434", # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    api_key="YOUR_API_KEY" # replace with API key if necessary
    num_ctx=8192 # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```
</hfoption>
<hfoption id="Azure OpenAI">

To connect to Azure OpenAI, you can either use `AzureOpenAIServerModel` directly, or use `LiteLLMModel` and configure it accordingly.

To initialize an instance of `AzureOpenAIServerModel`, you need to pass your model deployment name and then either pass the `azure_endpoint`, `api_key`, and `api_version` arguments, or set the environment variables `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `OPENAI_API_VERSION`.

```python
# !pip install smolagents[openai]
from smolagents import CodeAgent, AzureOpenAIServerModel

model = AzureOpenAIServerModel(model_id="gpt-4o-mini")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
```

Similarly, you can configure `LiteLLMModel` to connect to Azure OpenAI as follows:

- pass your model deployment name as `model_id`, and make sure to prefix it with `azure/`
- make sure to set the environment variable `AZURE_API_VERSION`
- either pass the `api_base` and `api_key` arguments, or set the environment variables `AZURE_API_KEY`, and `AZURE_API_BASE`

```python
import os
from smolagents import CodeAgent, LiteLLMModel

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-35-turbo-16k-deployment" # example of deployment name

os.environ["AZURE_API_KEY"] = "" # api_key
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2024-10-01-preview"

model = LiteLLMModel(model_id="azure/" + AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
   "Could you give me the 118th number in the Fibonacci sequence?",
)
```

</hfoption>
</hfoptions>

#### CodeAgent and ToolCallingAgent

The [`CodeAgent`] is our default agent. It will write and execute python code snippets at each step.

By default, the execution is done in your local environment.
This should be safe because the only functions that can be called are the tools you provided (especially if it's only tools by Hugging Face) and a set of predefined safe functions like `print` or functions from the `math` module, so you're already limited in what can be executed.

The Python interpreter also doesn't allow imports by default outside of a safe list, so all the most obvious attacks shouldn't be an issue.
You can authorize additional imports by passing the authorized modules as a list of strings in argument `additional_authorized_imports` upon initialization of your [`CodeAgent`]:

```py
model = HfApiModel()
agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

> [!WARNING]
> The LLM can generate arbitrary code that will then be executed: do not add any unsafe imports!

The execution will stop at any code trying to perform an illegal operation or if there is a regular Python error with the code generated by the agent.

You can also use [E2B code executor](https://e2b.dev/docs#what-is-e2-b) instead of a local Python interpreter by first [setting the `E2B_API_KEY` environment variable](https://e2b.dev/dashboard?tab=keys) and then passing `use_e2b_executor=True` upon agent initialization.

> [!TIP]
> Learn more about code execution [in this tutorial](tutorials/secure_code_execution).

We also support the widely-used way of writing actions as JSON-like blobs: this is [`ToolCallingAgent`], it works much in the same way like [`CodeAgent`], of course without `additional_authorized_imports` since it doesn't execute code:

```py
from smolagents import ToolCallingAgent

agent = ToolCallingAgent(tools=[], model=model)
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

### Inspecting an agent run

Here are a few useful attributes to inspect what happened after a run:
- `agent.logs` stores the fine-grained logs of the agent. At every step of the agent's run, everything gets stored in a dictionary that then is appended to `agent.logs`.
- Running `agent.write_memory_to_messages()` writes the agent's memory as list of chat messages for the Model to view. This method goes over each step of the log and only stores what it's interested in as a message: for instance, it will save the system prompt and task in separate messages, then for each step it will store the LLM output as a message, and the tool call output as another message. Use this if you want a higher-level view of what has happened - but not every log will be transcripted by this method.

## Tools

A tool is an atomic function to be used by an agent. To be used by an LLM, it also needs a few attributes that constitute its API and will be used to describe to the LLM how to call this tool:
- A name
- A description
- Input types and descriptions
- An output type

You can for instance check the [`PythonInterpreterTool`]: it has a name, a description, input descriptions, an output type, and a `forward` method to perform the action.

When the agent is initialized, the tool attributes are used to generate a tool description which is baked into the agent's system prompt. This lets the agent know which tools it can use and why.

### Default toolbox

Transformers comes with a default toolbox for empowering agents, that you can add to your agent upon initialization with argument `add_base_tools = True`:

- **DuckDuckGo web search***: performs a web search using DuckDuckGo browser.
- **Python code interpreter**: runs your LLM generated Python code in a secure environment. This tool will only be added to [`ToolCallingAgent`] if you initialize it with `add_base_tools=True`, since code-based agent can already natively execute Python code
- **Transcriber**: a speech-to-text pipeline built on Whisper-Turbo that transcribes an audio to text.

You can manually use a tool by calling it with its arguments.

```python
from smolagents import DuckDuckGoSearchTool

search_tool = DuckDuckGoSearchTool()
print(search_tool("Who's the current president of Russia?"))
```

### Create a new tool

You can create your own tool for use cases not covered by the default tools from Hugging Face.
For example, let's create a tool that returns the most downloaded model for a given task from the Hub.

You'll start with the code below.

```python
from huggingface_hub import list_models

task = "text-classification"

most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(most_downloaded_model.id)
```

This code can quickly be converted into a tool, just by wrapping it in a function and adding the `tool` decorator:
This is not the only way to build the tool: you can directly define it as a subclass of [`Tool`], which gives you more flexibility, for instance the possibility to initialize heavy class attributes.

Let's see how it works for both options:

<hfoptions id="build-a-tool">
<hfoption id="Decorate a function with @tool">

```py
from smolagents import tool

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return most_downloaded_model.id
```

The function needs:
- A clear name. The name should be descriptive enough of what this tool does to help the LLM brain powering the agent. Since this tool returns the model with the most downloads for a task, let's name it `model_download_tool`.
- Type hints on both inputs and output
- A description, that includes an 'Args:' part where each argument is described (without a type indication this time, it will be pulled from the type hint). Same as for the tool name, this description is an instruction manual for the LLM powering you agent, so do not neglect it.
All these elements will be automatically baked into the agent's system prompt upon initialization: so strive to make them as clear as possible!

> [!TIP]
> This definition format is the same as tool schemas used in `apply_chat_template`, the only difference is the added `tool` decorator: read more on our tool use API [here](https://huggingface.co/blog/unified-tool-use#passing-tools-to-a-chat-template).
</hfoption>
<hfoption id="Subclass Tool">

```py
from smolagents import Tool

class ModelDownloadTool(Tool):
    name = "model_download_tool"
    description = "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint."
    inputs = {"task": {"type": "string", "description": "The task for which to get the download count."}}
    output_type = "string"

    def forward(self, task: str) -> str:
        most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return most_downloaded_model.id
```

The subclass needs the following attributes:
- A clear `name`. The name should be descriptive enough of what this tool does to help the LLM brain powering the agent. Since this tool returns the model with the most downloads for a task, let's name it `model_download_tool`.
- A `description`. Same as for the `name`, this description is an instruction manual for the LLM powering you agent, so do not neglect it.
- Input types and descriptions
- Output type
All these attributes will be automatically baked into the agent's system prompt upon initialization: so strive to make them as clear as possible!
</hfoption>
</hfoptions>


Then you can directly initialize your agent:
```py
from smolagents import CodeAgent, HfApiModel
agent = CodeAgent(tools=[model_download_tool], model=HfApiModel())
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```

You get the following logs:
```text
╭──────────────────────────────────────── New run ─────────────────────────────────────────╮
│                                                                                          │
│ Can you give me the name of the model that has the most downloads in the 'text-to-video' │
│ task on the Hugging Face Hub?                                                            │
│                                                                                          │
╰─ HfApiModel - Qwen/Qwen2.5-Coder-32B-Instruct ───────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 model_name = model_download_tool(task="text-to-video")                               │
│   2 print(model_name)                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Execution logs:
ByteDance/AnimateDiff-Lightning

Out: None
[Step 0: Duration 0.27 seconds| Input tokens: 2,069 | Output tokens: 60]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╭─ Executing this code: ───────────────────────────────────────────────────────────────────╮
│   1 final_answer("ByteDance/AnimateDiff-Lightning")                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
Out - Final answer: ByteDance/AnimateDiff-Lightning
[Step 1: Duration 0.10 seconds| Input tokens: 4,288 | Output tokens: 148]
Out[20]: 'ByteDance/AnimateDiff-Lightning'
```

> [!TIP]
> Read more on tools in the [dedicated tutorial](./tutorials/tools#what-is-a-tool-and-how-to-build-one).

## Multi-agents

Multi-agent systems have been introduced with Microsoft's framework [Autogen](https://huggingface.co/papers/2308.08155).

In this type of framework, you have several agents working together to solve your task instead of only one.
It empirically yields better performance on most benchmarks. The reason for this better performance is conceptually simple: for many tasks, rather than using a do-it-all system, you would prefer to specialize units on sub-tasks. Here, having agents with separate tool sets and memories allows to achieve efficient specialization. For instance, why fill the memory of the code generating agent with all the content of webpages visited by the web search agent? It's better to keep them separate.

You can easily build hierarchical multi-agent systems with `smolagents`.

To do so, encapsulate the agent in a [`ManagedAgent`] object. This object needs arguments `agent`, `name`, and a `description`, which will then be embedded in the manager agent's system prompt to let it know how to call this managed agent, as we also do for tools.

Here's an example of making an agent that managed a specific web search agent using our [`DuckDuckGoSearchTool`]:

```py
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, ManagedAgent

model = HfApiModel()

web_agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="web_search",
    description="Runs web searches for you. Give it your query as an argument."
)

manager_agent = CodeAgent(
    tools=[], model=model, managed_agents=[managed_web_agent]
)

manager_agent.run("Who is the CEO of Hugging Face?")
```

> [!TIP]
> For an in-depth example of an efficient multi-agent implementation, see [how we pushed our multi-agent system to the top of the GAIA leaderboard](https://huggingface.co/blog/beating-gaia).


## Talk with your agent and visualize its thoughts in a cool Gradio interface

You can use `GradioUI` to interactively submit tasks to your agent and observe its thought and execution process, here is an example:

```py
from smolagents import (
    load_tool,
    CodeAgent,
    HfApiModel,
    GradioUI
)

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

model = HfApiModel(model_id)

# Initialize the agent with the image generation tool
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()
```

Under the hood, when the user types a new answer, the agent is launched with `agent.run(user_request, reset=False)`.
The `reset=False` flag means the agent's memory is not flushed before launching this new task, which lets the conversation go on.

You can also use this `reset=False` argument to keep the conversation going in any other agentic application.

## Next steps

For more in-depth usage, you will then want to check out our tutorials:
- [the explanation of how our code agents work](./tutorials/secure_code_execution)
- [this guide on how to build good agents](./tutorials/building_good_agents).
- [the in-depth guide for tool usage](./tutorials/building_good_agents).
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Agentic RAG

[[open-in-colab]]

Retrieval-Augmented-Generation (RAG) is “using an LLM to answer a user query, but basing the answer on information retrieved from a knowledge base”. It has many advantages over using a vanilla or fine-tuned LLM: to name a few, it allows to ground the answer on true facts and reduce confabulations, it allows to provide the LLM with domain-specific knowledge, and it allows fine-grained control of access to information from the knowledge base.

But vanilla RAG has limitations, most importantly these two:
- It performs only one retrieval step: if the results are bad, the generation in turn will be bad.
- Semantic similarity is computed with the user query as a reference, which might be suboptimal: for instance, the user query will often be a question and the document containing the true answer will be in affirmative voice, so its similarity score will be downgraded compared to other source documents in the interrogative form, leading to a risk of missing the relevant information.

We can alleviate these problems by making a RAG agent: very simply, an agent armed with a retriever tool!

This agent will: ✅ Formulate the query itself and ✅ Critique to re-retrieve if needed.

So it should naively recover some advanced RAG techniques!
- Instead of directly using the user query as the reference in semantic search, the agent formulates itself a reference sentence that can be closer to the targeted documents, as in [HyDE](https://huggingface.co/papers/2212.10496).
The agent can use the generated snippets and re-retrieve if needed, as in [Self-Query](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/).

Let's build this system. 🛠️

Run the line below to install required dependencies:
```bash
!pip install smolagents pandas langchain langchain-community sentence-transformers datasets python-dotenv rank_bm25 --upgrade -q
```
To call the HF Inference API, you will need a valid token as your environment variable `HF_TOKEN`.
We use python-dotenv to load it.
```py
from dotenv import load_dotenv
load_dotenv()
```

We first load a knowledge base on which we want to perform RAG: this dataset is a compilation of the documentation pages for many Hugging Face libraries, stored as markdown. We will keep only the documentation for the `transformers` library.

Then prepare the knowledge base by processing the dataset and storing it into a vector database to be used by the retriever.

We use [LangChain](https://python.langchain.com/docs/introduction/) for its excellent vector database utilities.

```py
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)
```

Now the documents are ready.

So let’s build our agentic RAG system!

👉 We only need a RetrieverTool that our agent can leverage to retrieve information from the knowledge base.

Since we need to add a vectordb as an attribute of the tool, we cannot simply use the simple tool constructor with a `@tool` decorator: so we will follow the advanced setup highlighted in the [tools tutorial](../tutorials/tools).

```py
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

retriever_tool = RetrieverTool(docs_processed)
```
We have used BM25, a classic retrieval method, because it's lightning fast to setup.
To improve retrieval accuracy, you could use replace BM25 with semantic search using vector representations for documents: thus you can head to the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) to select a good embedding model.

Now it’s straightforward to create an agent that leverages this `retriever_tool`!

The agent will need these arguments upon initialization:
- `tools`: a list of tools that the agent will be able to call.
- `model`: the LLM that powers the agent.
Our `model` must be a callable that takes as input a list of messages and returns text. It also needs to accept a stop_sequences argument that indicates when to stop its generation. For convenience, we directly use the HfEngine class provided in the package to get a LLM engine that calls Hugging Face's Inference API.

>[!NOTE] To use a specific model, pass it like this: `HfApiModel("meta-llama/Llama-3.3-70B-Instruct")`. The Inference API hosts models based on various criteria, and deployed models may be updated or replaced without prior notice. Learn more about it [here](https://huggingface.co/docs/api-inference/supported-models).

```py
from smolagents import HfApiModel, CodeAgent

agent = CodeAgent(
    tools=[retriever_tool], model=HfApiModel(), max_steps=4, verbosity_level=2
)
```
Upon initializing the CodeAgent, it has been automatically given a default system prompt that tells the LLM engine to process step-by-step and generate tool calls as code snippets, but you could replace this prompt template with your own as needed.

Then when its `.run()` method is launched, the agent takes care of calling the LLM engine, and executing the tool calls, all in a loop that ends only when tool `final_answer` is called with the final answer as its argument.

```py
agent_output = agent.run("For a transformers model training, which is slower, the forward or the backward pass?")

print("Final output:")
print(agent_output)
```





