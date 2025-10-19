import asyncio
import json
import logging
import os
import random
import textwrap
from typing import Any, Callable, Dict

import httpx

logger = logging.getLogger(__name__)


# --- This class is now LLM-powered (OpenAI) ---
class TestAgent:
    # conversation_history: List[dict]  # [{'user': ...}, {'assistant': ...}, ...]
    """
    Represents a generic, domain-agnostic testing agent that simulates a user.

    It holds a "persona" (the desired outcome) and interacts with any
    provided "tool" (as an async function) to achieve its goal using an LLM.
    
    The persona dictionary dictates all domain-specific behavior.
    """

    async def safe_tool_to_test(self, query):
        result = await self.app_tool(query)
        if isinstance(result, dict):
            return result
        return {"response": result}
    
    def __init__(self, tool_to_test: Callable, user_persona: Dict[str, Any]):
        """
        Initializes the testing agent.
        
        Args:
            tool_to_test: An async function (or method) to be tested.
                          It MUST accept two arguments:
                          1. query (str): The user's input.
                          2. context (Dict[str, Any]): The current conversation state.
                          It MUST return a dictionary with at least:
                          - "status": (str) e.g., "complete", "clarification_needed"
                          - "response": (str) The application's reply.
                          - "context": (Dict[str, Any]) The updated conversation state.
            
            user_persona: A dict representing the user's goals and behavior.
                          It MUST contain the following keys:
                          - "data": (Dict[str, Any]) The key-value data for the persona 
                            (e.g., {"destination": "Paris", "passengers": 2}).
                          - "goal": (str) The overall objective 
                            (e.g., "Book a flight to Paris for 2 people tomorrow.").
                          - "initial_query_prompt": (str) A system prompt template for 
                            generating the *first* message. Can use {goal} and {persona_data}
                            as format variables.
                          - "clarification_prompt": (str) A system prompt template for
                            answering clarification questions. Can use {goal} and 
                            {persona_data} as format variables.
        """
        if not asyncio.iscoroutinefunction(tool_to_test):
            raise TypeError("The 'tool_to_test' must be an async function (defined with 'async def').")
            
        self.app_tool = tool_to_test
        self.persona = user_persona
        self.conversation_history = []  # [{'user': ...}, {'assistant': ...}, ...]

        # Validate persona structure
        required_keys = ["data", "goal", "initial_query_prompt", "clarification_prompt"]
        if not all(key in self.persona for key in required_keys):
            missing = [key for key in required_keys if key not in self.persona]
            raise ValueError(f"user_persona is missing required keys: {missing}")

        # Get API key from environment
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.info("--- [TestAgent ERROR] OPENAI_API_KEY environment variable not set. LLM calls will FAIL. ---")

        logger.info(f"TestAgent initialized with persona goal: {self.persona.get('goal')}")

    async def _call_openai_api(self, system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
        """Calls the OpenAI API with exponential backoff."""
        
        # --- Removed Mocking Logic ---
        if not self.api_key:
            logger.info("   [TestAgent LLM ERROR] API key not set. Cannot generate simulated answer.")
            return "Error: OPENAI_API_KEY is missing."
        # --- End Removed Mocking Logic ---

        # OpenAI API endpoint
        api_url = "https://api.openai.com/v1/chat/completions"

        # OpenAI-specific payload
        payload = {
            "model": "gpt-4o", # You can change this to gpt-4, gpt-4o, etc.
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.01,
            "top_p": 1,
            "max_tokens": 2000, # OpenAI uses max_tokens, not maxOutputTokens
        }
        
        # OpenAI uses Bearer token authentication
        headers = { 
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        base_delay_s = 1
        
        # Use httpx.AsyncClient for async requests
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(max_retries):
                try:
                    logger.info(f"   [TestAgent LLM Call] Attempt {attempt+1} to OpenAI API...")
                    response = await client.post(api_url, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Extract text from the OpenAI response
                        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        if text:
                            logger.info("   [TestAgent LLM Success] Received response.")
                            return text.strip()
                        else:
                            logger.info(f"   [TestAgent LLM WARNING] Received 200 OK but no text in response: {result}")
                            # Fallthrough to retry logic
                    
                    elif response.status_code == 429 or response.status_code >= 500:
                        # Throttling or server error, time to back off
                        logger.info(f"   [TestAgent LLM Retryable Error] Status {response.status_code}. Retrying...")
                    
                    else:
                        # Client-side error (4xx) that isn't 429. Don't retry.
                        logger.info(f"   [TestAgent LLM Non-Retryable Error] Status {response.status_code}: {response.text}")
                        return f"Error: Failed to call LLM with status {response.status_code}"

                except httpx.RequestError as e:
                    logger.info(f"   [TestAgent LLM Request Error] An error occurred while requesting {e.request.url!r}: {e}")
                    # This is likely retryable (e.g., connection error)
                
                # If we're here, we need to retry
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = (base_delay_s * (2 ** attempt)) + (random.uniform(0, 1))
                    logger.info(f"   [TestAgent LLM Retry] Waiting {delay:.2f}s before next attempt...")
                    await asyncio.sleep(delay)
                
        logger.info(f"   [TestAgent LLM ERROR] Failed to get response after {max_retries} attempts.")
        return "Error: LLM call failed after all retries."
        
    async def _generate_initial_query(self) -> str:
        """
        Generates the first message based on the TestAgent's persona and goal.
        The prompt logic is defined *in the persona*.
        """
        persona_data_str = json.dumps(self.persona.get("data", {}))
        persona_goal = self.persona.get("goal", "No goal specified.")
        
        # Get the prompt template *from the persona*
        try:
            system_prompt_template = self.persona.get("initial_query_prompt", "")
            system_prompt = system_prompt_template.format(
                goal=persona_goal,
                persona_data=persona_data_str
            )
        except (KeyError, TypeError) as e:
            logger.info(f"   [TestAgent ERROR] Invalid 'initial_query_prompt' template: {e}")
            return "Error: Invalid initial_query_prompt in persona."

        # The user prompt is generic, asking the LLM to execute its instructions
        user_prompt = "Please generate the initial message now."
        
        logger.info("   [TestAgent LLM Call] Generating initial query...")
        initial_query = await self._call_openai_api(system_prompt, user_prompt)
        if initial_query.startswith("Error:"):
            # Handle error from LLM call
            logger.info(f"   [TestAgent ERROR] Failed to generate initial query: {initial_query}")
            return "Error: Failed to generate initial query."
        return initial_query


    async def _simulate_user_answer_llm(self, question: str) -> str:
        """
        Simulates a user's response based on the persona using an LLM.
        The prompt logic is defined *in the persona*.
        """
        
        # 1. Build the System Prompt (from the persona)
        persona_data_str = json.dumps(self.persona.get("data", {}))
        persona_goal = self.persona.get("goal", "No goal specified.")

        try:
            system_prompt_template = self.persona.get("clarification_prompt", "")
            system_prompt = system_prompt_template.format(
                goal=persona_goal,
                persona_data=persona_data_str
            )
        except (KeyError, TypeError) as e:
            logger.info(f"   [TestAgent ERROR] Invalid 'clarification_prompt' template: {e}")
            return "Error: Invalid clarification_prompt in persona."

        # 2. Build the User Prompt (the input to the LLM) - Always use conversation_history
        history_str = ""
        for turn in self.conversation_history:
            if 'user' in turn:
                history_str += f"User: {turn['user']}\n"
            elif 'assistant' in turn:
                history_str += f"Agent: {turn['assistant']}\n"

        user_prompt = textwrap.dedent(f"""
            Here is the goal you are trying to achieve:
            <goal>
            {persona_goal}
            </goal>

            Here is the conversation so far:
            <history>
            {history_str.strip()}
            </history>

            The agent just asked you this question:
            <question>
            {question}
            </question>

            Based on your instructions and the conversation so far, either answer the question or indicate that the conversation is complete if the user's goal is achieved.
        """).strip()

        # 3. Call the LLM
        response_text = await self._call_openai_api(system_prompt, user_prompt)

        if response_text.startswith("Error:"):
            return None # Propagate failure

        return response_text

    async def run_test(self, max_turns: int = 5) -> list:
        """
        Runs a full test simulation.
        
        The TestAgent generates the initial query based on its persona,
        making it agnostic to the exact first message.
        
        Args:
            max_turns: A safeguard against infinite loops.
        Returns:
            The conversation history (list of dicts)
        """
        logger.info("\n--- Starting New Test Simulation ---")
        logger.info(f"Goal: {self.persona.get('goal', 'Complete booking')}")
        
        self.conversation_history = [] # Reset history for new test

        # The TestAgent generates the initial query
        current_query = await self._generate_initial_query()
        if current_query.startswith("Error:"):
            logger.info("   [TestAgent FATAL] Could not generate initial query.")
            return self.conversation_history # Fail fast if LLM fails

        logger.info(f"Initial Query (Generated): '{current_query}'")

        for turn in range(max_turns):
            logger.info(f"\n--- Turn {turn + 1} ---")
            # 1. TestAgent "speaks" to the AppAgent (its tool)
            logger.info(f"[TestAgent -> AppAgent]: {current_query}")
            self.conversation_history.append({'user': current_query})
            # 2. AppAgent processes the request (calling the generic tool)
            try:
                app_response = await self.safe_tool_to_test(current_query)
            except Exception as e:
                logger.info("\n--- Test Failed ---")
                logger.info(f"The application tool itself raised an exception: {e}")
                return self.conversation_history
            # 3. TestAgent "hears" the AppAgent's response
            logger.info(f"[AppAgent -> TestAgent]: {json.dumps(app_response, indent=2)}")
            # Store the *dictionary* response for full context
            response_text = app_response.get("response")
            self.conversation_history.append({'assistant': response_text})

            # 4. TestAgent analyzes the response
            # Let the LLM decide if the conversation is complete or needs clarification
            simulated_answer = await self._simulate_user_answer_llm(response_text)
            if simulated_answer is None or (isinstance(simulated_answer, str) and simulated_answer.startswith("Error:")):
                logger.info("\n--- Test Ended ---")
                logger.info(f"TestAgent (LLM) did not generate a further answer. Final result: {response_text}")
                return self.conversation_history
            # Heuristic: If the LLM says the conversation is complete, end
            if any(phrase in simulated_answer.lower() for phrase in ["conversation is complete", "booking complete", "goal achieved", "no further action", "thank you", "done"]):
                logger.info("\n--- Test Succeeded ---")
                logger.info(f"Final Result: {response_text}")
                return self.conversation_history
            current_query = simulated_answer
        logger.info(f"\n--- Test Ended: Max turns ({max_turns}) reached ---")
        return self.conversation_history

