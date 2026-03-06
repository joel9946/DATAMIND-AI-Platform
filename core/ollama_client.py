"""
╔══════════════════════════════════════════════════════════════════╗
║  ollama_client.py  —  DataMind Platform                         ║
║  THE BRAIN TALKER                                                ║
║                                                                  ║
║  This file is the messenger between DataMind and the AI brain.  ║
║  Imagine you want to ask a question to a genius locked in a     ║
║  room. You can't go in directly — you slide a note under the    ║
║  door. This file writes those notes and reads the answers.      ║
║                                                                  ║
║  The "genius in the room" is Ollama — an AI program running     ║
║  on YOUR computer. We talk to it by sending HTTP messages       ║
║  (like mini web requests) to a local address.                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
# Think of imports like picking tools out of a toolbox before you start work.

import requests   # The tool for sending messages over the internet (or local network).
                  # Like a postman that carries our questions to Ollama.

import json       # Ollama speaks in "JSON" format — it's like a structured letter format.
                  # e.g. {"question": "hello", "answer": "hi there"}
                  # json.loads() reads those letters. json.dumps() writes them.

import time       # Lets us pause or measure how long things take.
                  # Like a stopwatch — we can say "wait 2 seconds".

from typing import Generator, Optional, List, Dict, Any
# These are "type hints" — they tell other programmers what kind of data
# a function expects and returns. It's like labelling boxes before packing.
# Generator = a function that gives you one result at a time (like a vending machine)
# Optional  = the value might be there, or it might be None (empty)
# List      = a list of things   e.g. ["apple", "banana"]
# Dict      = a dictionary       e.g. {"name": "Ali", "age": 10}
# Any       = could be anything — we don't care about the type


# ── CONSTANT ─────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
# This is the "address" of Ollama on your computer.
# "localhost" means "this very computer, not the internet".
# "11434" is the port number — like a door number in an apartment building.
# So 11434 is the specific door Ollama is listening at.


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS: OllamaClient
#  A "class" is like a blueprint for making objects.
#  Like a blueprint for a phone — once you have the blueprint,
#  you can make as many phone objects as you need.
#  OllamaClient is our "phone" for calling Ollama.
# ══════════════════════════════════════════════════════════════════════════════

class OllamaClient:
    """
    A class that handles ALL communication with the Ollama AI.

    Think of it like a phone operator: it manages the connection
    to your local LLM, routes your messages, and streams
    responses back word-by-word for a real-time feel.
    """

    # ── __init__ : The Constructor ────────────────────────────────────────────
    # __init__ runs automatically when you create a new OllamaClient object.
    # It sets up the object, like charging a phone before using it.

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 120):
        # base_url : where to find Ollama (default = localhost:11434)
        # timeout  : how many seconds to wait before giving up (default = 120 seconds)

        self.base_url = base_url
        # "self" means "store this on THIS object". Like writing your name
        # inside your book so you know it's yours.

        self.timeout = timeout
        # Store the timeout value so every function in this class can use it.

        self.session = requests.Session()
        # A Session is like keeping a phone line open instead of dialling
        # fresh every single time. It's faster — we reuse the same connection.

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1: Model Management
    #  Functions for checking what AI models are available and if Ollama is on.
    # ══════════════════════════════════════════════════════════════════════════

    def list_models(self) -> List[str]:
        """Returns a list of locally available Ollama models."""
        # This asks Ollama: "Hey, which AI models do you have installed?"
        try:
            # try: attempt this code. If something goes wrong, jump to "except".

            r = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            # .get() sends a GET request — like asking "show me your menu".
            # /api/tags is the specific page on Ollama that lists installed models.
            # timeout=5 means: if no reply in 5 seconds, stop waiting.
            # The result is stored in "r" (short for "response").

            r.raise_for_status()
            # If Ollama replied with an error (like "not found"), this line
            # raises an alarm so we jump to the except block.
            # Like checking if the postman actually delivered the letter.

            return [m["name"] for m in r.json().get("models", [])]
            # r.json()        — convert the response from JSON text to a Python dict
            # .get("models", []) — get the "models" list; if missing, use empty list []
            # [m["name"] for m in ...]  — loop through every model, grab just its name
            # This is called a "list comprehension" — a fast way to build a list in one line.

        except Exception:
            # If ANY error happened (Ollama is off, network error, etc.)
            # we just return an empty list instead of crashing the whole app.
            return []

    def is_online(self) -> bool:
        """Pings Ollama to check if it's running. Returns True or False."""
        # Like knocking on a door to see if anyone is home.
        try:
            r = self.session.get(f"{self.base_url}/", timeout=3)
            # Ping the home page of Ollama. timeout=3 is short — just a quick knock.

            return r.status_code == 200
            # HTTP status 200 means "OK / success".
            # So we return True if status=200, False otherwise.
            # Like: "Did the door open? Yes → True. No → False."

        except Exception:
            return False
            # Any error means Ollama is definitely not running → return False.

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Fetches metadata about a specific model (size, parameters, etc.)."""
        try:
            r = self.session.post(
                f"{self.base_url}/api/show",   # The URL that gives model info
                json={"name": model},           # Send the model name as a JSON body
                timeout=5
            )
            return r.json()
            # Return all the info Ollama sent back as a Python dictionary.

        except Exception:
            return {}
            # If something fails, return an empty dict (no info).

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2: Complete (non-streaming)
    #  Sends a question, waits for the FULL answer, returns it.
    #  Good for background tasks where you don't need word-by-word output.
    # ══════════════════════════════════════════════════════════════════════════

    def complete(
        self,
        prompt: str,           # The question or instruction to send to the AI
        model: str,            # Which AI model to use (e.g. "llama3.2")
        system: Optional[str] = None,  # Optional personality instructions for the AI
        temperature: float = 0.7,      # How creative the AI should be (0=robotic, 1=creative)
        max_tokens: int = 2048,        # Maximum length of the answer (in "tokens" ≈ words)
    ) -> str:
        """
        Sends a single prompt and waits for the full response.
        Best for background tasks where you don't need real-time output.
        """
        payload = {
            "model": model,          # Tell Ollama which AI brain to use
            "prompt": prompt,        # The actual question or task
            "stream": False,         # False means: wait for the WHOLE answer, then send it
            "options": {
                "temperature": temperature,  # Creativity level (0.7 = balanced)
                "num_predict": max_tokens,   # Max number of tokens (word-pieces) to generate
            }
        }
        # payload is a dictionary — a structured "letter" we'll send to Ollama.

        if system:
            payload["system"] = system
            # If a system prompt was provided (e.g. "You are a data analyst"),
            # add it to the payload. This is like giving the AI a job description.

        try:
            r = self.session.post(
                f"{self.base_url}/api/generate",  # The Ollama endpoint for text generation
                json=payload,                     # Send our payload as JSON
                timeout=self.timeout              # Wait up to 120 seconds
            )
            r.raise_for_status()
            # Throw an error if Ollama returned a failure status code.

            return r.json().get("response", "")
            # r.json()            — parse the JSON response into a Python dict
            # .get("response", "") — get the "response" key; default to "" if missing

        except requests.exceptions.Timeout:
            return "[Error] Request timed out. The model might be slow or unavailable."
            # This specific error happens when Ollama took too long to reply.

        except Exception as e:
            return f"[Error] {str(e)}"
            # Catch any other unexpected error and return it as a string message.

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 3: Stream
    #  Sends a question and gets the answer WORD BY WORD as it's generated.
    #  This creates the "typewriter" effect in the UI — you see the AI thinking!
    # ══════════════════════════════════════════════════════════════════════════

    def stream(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        Streams the LLM response token-by-token.
        A Generator function uses 'yield' — it pauses and hands you
        one value at a time, instead of waiting to return everything at once.

        Like a vending machine that gives you one crisp at a time vs
        dumping the whole bag in your lap.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,   # ← KEY DIFFERENCE: True means send tokens one-by-one
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        if system:
            payload["system"] = system

        try:
            with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,         # Tell the requests library to stream the HTTP response
                timeout=self.timeout
            ) as r:
                # "with" is a context manager — it automatically closes the connection
                # when we're done, like automatically turning off a tap when you leave the kitchen.

                r.raise_for_status()

                for line in r.iter_lines():
                    # r.iter_lines() reads the response one line at a time.
                    # Each line from Ollama is one token (a word or part of a word).
                    # Like reading a book line by line instead of all at once.

                    if line:
                        # Skip empty lines (like skipping blank pages in a book).
                        try:
                            chunk = json.loads(line)
                            # Each line is a JSON string. Parse it into a dictionary.
                            # e.g. '{"response": "Hello", "done": false}'
                            # becomes: {"response": "Hello", "done": False}

                            token = chunk.get("response", "")
                            # Get the actual text token from this chunk.
                            # e.g. "Hello" or " world" or "!"

                            if token:
                                yield token
                                # YIELD: pause this function and hand this token
                                # to whoever is calling stream().
                                # The caller shows it on screen, then we resume.

                            if chunk.get("done", False):
                                break
                                # If Ollama says "done: true", the answer is complete.
                                # Break out of the for loop — no more tokens coming.

                        except json.JSONDecodeError:
                            continue
                            # If a line isn't valid JSON, skip it and continue.

        except requests.exceptions.Timeout:
            yield "\n[Error] Request timed out."
            # Yield an error message as the last token if we ran out of time.

        except Exception as e:
            yield f"\n[Error] {str(e)}"
            # Yield any other error as a message token.

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 4: Multi-turn Chat
    #  Like a proper conversation — the AI REMEMBERS what was said before.
    #  We send the full history each time so the AI has context.
    # ══════════════════════════════════════════════════════════════════════════

    def chat(
        self,
        messages: List[Dict[str, str]],
        # messages is a list of conversation turns, e.g.:
        # [{"role": "user", "content": "Hello"},
        #  {"role": "assistant", "content": "Hi! How can I help?"},
        #  {"role": "user", "content": "What is 2+2?"}]
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = True,  # Default is streaming (word by word)
    ) -> Generator[str, None, None]:
        """
        Full multi-turn conversation using Ollama's /api/chat endpoint.
        Preserves history so the AI remembers earlier messages.
        """
        payload = {
            "model": model,
            "messages": messages,   # The full conversation history
            "stream": stream,
            "options": {"temperature": temperature}
        }

        if system:
            # If a system prompt is given, stick it at the BEGINNING of the messages list.
            # This is like giving the AI a job description before the conversation starts.
            payload["messages"] = [{"role": "system", "content": system}] + messages
            # The "+" operator joins two lists together.

        try:
            with self.session.post(
                f"{self.base_url}/api/chat",   # Different endpoint: /api/chat (not /api/generate)
                json=payload,
                stream=stream,
                timeout=self.timeout
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)

                            token = chunk.get("message", {}).get("content", "")
                            # Chat responses have a different structure than generate responses.
                            # The token is nested: chunk → "message" → "content"
                            # .get("message", {}) gets "message" or empty dict if missing.
                            # .get("content", "") then gets "content" from that.

                            if token:
                                yield token

                            if chunk.get("done", False):
                                break

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            yield f"\n[Error] {str(e)}"

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 5: Specialised AI Agents
    #  Each function below is a different "personality" for the AI.
    #  Same Ollama brain, but different system prompts = different expertise.
    #  Like one person wearing different hats: doctor hat, chef hat, lawyer hat.
    # ══════════════════════════════════════════════════════════════════════════

    def analyze_data(self, df_summary: str, question: str, model: str) -> Generator[str, None, None]:
        """
        Agent 1: The Data Analyst
        Wears the "expert data scientist" hat.
        Given a summary of the dataset and a question, it analyses and explains.
        """
        system = """You are DataMind's expert data analyst AI. Your role is to:
1. Analyze datasets and provide deep, actionable insights
2. Generate precise Python/Pandas code when asked
3. Identify patterns, anomalies, and statistical insights
4. Explain findings in a clear, structured way

Always format code in ```python blocks.
Be concise but thorough. Lead with the most important insight."""
        # This system prompt is like a job description pinned above the AI's desk.
        # It tells the AI exactly how to behave for this kind of question.

        prompt = f"""Dataset Summary:
{df_summary}

User Question: {question}

Provide a thorough analysis with code examples where appropriate."""
        # f-string: the {} parts get replaced with real values.
        # So {df_summary} becomes the actual dataset description text.

        yield from self.stream(prompt, model, system=system)
        # "yield from" forwards every yielded value from self.stream() to our caller.
        # Like a relay race: self.stream() passes the baton to analyze_data(),
        # which passes it straight to whoever called analyze_data().

    def generate_eda_insights(self, stats_json: str, model: str) -> str:
        """
        Agent 2: The EDA Narrator
        Takes raw statistics and writes a human-readable story about the data.
        Returns a full string (not streaming) because it's used in reports.
        """
        system = """You are a senior data scientist providing EDA (Exploratory Data Analysis) insights.
Analyze the provided statistics and generate 5-7 key insights about the dataset.
Format as a numbered list. Be specific with numbers. Identify any data quality issues."""

        prompt = f"""Descriptive Statistics:
{stats_json}

Generate key EDA insights from these statistics."""

        return self.complete(prompt, model, system=system)
        # Uses complete() (not stream) — we want the WHOLE answer at once,
        # not word by word, because we'll display it in a box later.

    def explain_model(self, model_name: str, metrics: Dict, feature_importance: Dict, model_llm: str) -> str:
        """
        Agent 3: The Model Explainer
        Takes ML model results (accuracy, feature importance etc.) and explains
        them in plain English — for non-technical stakeholders.
        Like a translator between "machine speak" and human speak.
        """
        system = """You are an ML engineer explaining a trained model to both technical and non-technical stakeholders.
Be clear, specific, and actionable. Include what the metrics mean in plain English."""

        prompt = f"""Model: {model_name}
Performance Metrics: {json.dumps(metrics, indent=2)}
Top Feature Importances: {json.dumps(feature_importance, indent=2)}

Write a comprehensive model explanation covering:
1. Model performance summary in plain English
2. What the key metrics mean for this use case
3. Which features drive predictions most and why
4. Recommendations for improving the model"""
        # json.dumps(metrics, indent=2) converts the metrics dictionary to a
        # nicely formatted JSON string. indent=2 adds 2 spaces of indentation
        # so it's readable. Like pretty-printing a receipt.

        return self.complete(prompt, model_llm, system=system)

    def nl_to_pandas(self, df_info: str, question: str, model: str) -> str:
        """
        Agent 4: The Code Translator
        Turns a plain-English question into executable Pandas code.
        e.g. "What is the average age?" → df['Age'].mean()

        This is the magic behind the "Data Chat" feature.
        """
        system = """You are a Python/Pandas expert. Convert natural language questions about dataframes into clean, executable Python code.

RULES:
- The dataframe variable is always named `df`
- Return ONLY the Python code, no explanation
- The code must end by storing the result in a variable called `result`
- Handle edge cases gracefully
- Use efficient Pandas methods"""
        # These rules are crucial! Without "return ONLY the Python code",
        # the AI might add explanations that break exec() when we run the code.

        prompt = f"""DataFrame Info:
{df_info}

Question: {question}

Generate Python/Pandas code to answer this question. Store the answer in `result`."""

        return self.complete(prompt, model, system=system, temperature=0.1)
        # temperature=0.1 means very LOW creativity.
        # For code generation, we want exact, reliable output — not creative variations.
        # 0.1 = "robotic and precise". This reduces AI hallucination in code.

    def suggest_features(self, df_summary: str, target_col: str, model: str) -> str:
        """
        Agent 5: The Feature Engineer
        Suggests ways to create new columns that might help ML model performance.
        e.g. "Combine Age and Class into an AgeClass ratio column."
        """
        system = "You are a senior ML feature engineer. Suggest creative and practical feature engineering ideas."

        prompt = f"""Dataset: {df_summary}
Target: {target_col}

Suggest 5 feature engineering ideas that could improve model performance."""

        return self.complete(prompt, model, system=system)


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLETON PATTERN
#  A Singleton means only ONE instance of OllamaClient ever exists.
#  Like a town having only one post office — everyone uses the same one.
#  This saves memory and keeps the HTTP connection (Session) alive efficiently.
# ══════════════════════════════════════════════════════════════════════════════

_client: Optional[OllamaClient] = None
# A module-level variable. Starts as None (empty).
# The underscore _ prefix means "this is private, don't use it directly from outside".

def get_client() -> OllamaClient:
    """
    Returns the ONE shared OllamaClient. Creates it if it doesn't exist yet.
    Every part of DataMind calls get_client() instead of creating a new one each time.
    """
    global _client
    # "global" means: I want to read/write the _client variable defined ABOVE this function,
    # not create a new local variable with the same name.

    if _client is None:
        # First time this is called? The post office doesn't exist yet. Build it.
        _client = OllamaClient()

    return _client
    # Return the existing (or newly created) client.
    # Second, third, fourth call → same object is returned every time.
