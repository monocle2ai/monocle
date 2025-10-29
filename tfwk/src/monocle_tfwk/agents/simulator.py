import asyncio
import json
import logging
import os
import random
import textwrap
from enum import StrEnum
from typing import Any, Callable, Dict

# --- Persona Schema ---
# The persona dictionary must have the following structure:
# {
#     "persona_data": Dict[str, Any],  # Key-value data for the persona (e.g., {"destination": "Paris", "passengers": 2})
#     "goal": str,  # The overall objective (e.g., "Book a flight to Paris for 2 people tomorrow.")
#     "initial_query_prompt": str,  # System prompt template for generating the first message. Can use {goal} and {persona_data} as format variables.
#     "clarification_prompt": str,  # System prompt template for answering clarification questions. Can use {goal} and {persona_data} as format variables.
# }

# --- Persona Required Keys Enum ---
class PersonaKeys(StrEnum):
    PERSONA_DATA = "persona_data"
    GOAL = "goal"
    INITIAL_QUERY_PROMPT = "initial_query_prompt"
    CLARIFICATION_PROMPT = "clarification_prompt"


# --- OpenAI API Configuration Constants ---
MODEL = "gpt-4o"  # You can change this to gpt-4, gpt-4o, etc.
API_URL = "https://api.openai.com/v1/chat/completions"
TEMPERATURE = 0.01
TOP_P = 1
MAX_TOKENS = 2000  # OpenAI uses max_tokens, not maxOutputTokens

logger = logging.getLogger(__name__)


# --- This class is now LLM-powered (OpenAI) ---
class TestAgent:

    DEFAULT_CLARIFICATION_PROMPT = textwrap.dedent("""
        You are a user answering a clarification question from an agent.
        Your goal is to provide answers to complete the clarification.
        You use the following persona data:
        {persona_data}

        You will be given the conversation history and a final question from the agent.
        Your job is to answer questions till the goal is achieved, using the information from your persona data or make up any missing details.
        - Do not be overly conversational.
        - Just provide the specific information requested.
    """).strip()
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
        """
        if not asyncio.iscoroutinefunction(tool_to_test):
            raise TypeError("The 'tool_to_test' must be an async function (defined with 'async def').")
            

        self.app_tool = tool_to_test
        # If clarification_prompt is missing, use the default
        persona = dict(user_persona)
        if PersonaKeys.CLARIFICATION_PROMPT not in persona:
            persona[PersonaKeys.CLARIFICATION_PROMPT] = self.DEFAULT_CLARIFICATION_PROMPT
        self.persona = persona
        self.conversation_history = []  # [{'user': ...}, {'assistant': ...}, ...]

        # Validate persona structure
        required_keys = [k.value for k in PersonaKeys]
        if not all(key in self.persona for key in required_keys):
            missing = [key for key in required_keys if key not in self.persona]
            raise ValueError(f"user_persona is missing required keys: {missing}")

        # Get API key from environment
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.info("--- [TestAgent ERROR] OPENAI_API_KEY environment variable not set. LLM calls will FAIL. ---")

        logger.info(f"TestAgent initialized with persona goal: {self.persona.get(PersonaKeys.GOAL)}")

    async def _call_openai_api(self, system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
        """Calls the OpenAI API using openai.OpenAI client, enforcing JSON output."""
        try:
            import openai
        except ImportError:
            logger.info("   [TestAgent LLM ERROR] openai package not installed.")
            return "Error: openai package not installed."
        if not self.api_key:
            logger.info("   [TestAgent LLM ERROR] API key not set. Cannot generate simulated answer.")
            return "Error: OPENAI_API_KEY is missing."

        client = openai.OpenAI(api_key=self.api_key)
        
        # --- FIX 1: Add JSON requirement to the prompt for API validation ---
        # This addresses the Error code 400: "'messages' must contain the word 'json' in some form..."
        json_enforcement_prompt = "\n\nYour final response MUST be a valid JSON object."
        modified_system_prompt = system_prompt + json_enforcement_prompt
        
        for attempt in range(max_retries):
            try:
                logger.info(f"   [TestAgent LLM Call] Attempt {attempt+1} to OpenAI API...")
                
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": modified_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"}
                )
                text = completion.choices[0].message.content.strip()
                if text:
                    logger.info("   [TestAgent LLM Success] Received response.")
                    return text
                else:
                    logger.info(f"   [TestAgent LLM WARNING] Received response but no text: {completion}")
            except Exception as e:
                logger.info(f"   [TestAgent LLM Error] {e}")
                if attempt < max_retries - 1:
                    delay = (1 * (2 ** attempt)) + (random.uniform(0, 1))
                    logger.info(f"   [TestAgent LLM Retry] Waiting {delay:.2f}s before next attempt...")
                    await asyncio.sleep(delay)
        logger.info(f"   [TestAgent LLM ERROR] Failed to get response after {max_retries} attempts.")
        return "Error: LLM call failed after all retries."
        
    async def _generate_initial_query(self) -> str:
        """
        Generates the first message based on the TestAgent's persona and goal.
        The prompt logic is defined *in the persona*.
        """
        persona_data_str = json.dumps(self.persona.get(PersonaKeys.PERSONA_DATA, {}))
        persona_goal = self.persona.get(PersonaKeys.GOAL, "No goal specified.")

        # Get the prompt template *from the persona*
        try:
            system_prompt_template = self.persona.get(PersonaKeys.INITIAL_QUERY_PROMPT, "")
            system_prompt = system_prompt_template.format(
                goal=persona_goal,
                persona_data=persona_data_str
            )
        except (KeyError, TypeError) as e:
            logger.info(f"   [TestAgent ERROR] Invalid '{PersonaKeys.INITIAL_QUERY_PROMPT}' template: {e}")
            return f"Error: Invalid {PersonaKeys.INITIAL_QUERY_PROMPT} in persona."

        # The user prompt is generic, asking the LLM to execute its instructions
        user_prompt = "Please generate the initial message now."
        logger.info("   [TestAgent LLM Call] Generating initial query...")
        initial_query = await self._call_openai_api(system_prompt, user_prompt)
        if initial_query.startswith("Error:"):
            # Handle error from LLM call
            logger.info(f"   [TestAgent ERROR] Failed to generate initial query: {initial_query}")
            return "Error: Failed to generate initial query."
        
        # NOTE: If this first call is *not* meant to return JSON,
        # you would need a separate LLM call function without the 
        # response_format={"type": "json_object"} parameter.
        # For now, we assume the initial query is a simple string the tool expects.
        try:
            # The initial query generator is often used to create a simple string.
            # We will attempt to parse the JSON and extract a 'response' key if it exists,
            # otherwise, we return the raw text (this is an assumption for 'gpt-4o' usage).
            result_dict = json.loads(initial_query)
            return result_dict.get("response", initial_query)
        except json.JSONDecodeError:
            return initial_query
        except AttributeError:
            # If the result isn't a string (shouldn't happen, but defensive)
            return str(initial_query)


    async def _simulate_user_answer_llm(self, question: str) -> dict:
        """
        Simulates a user's response based on the persona using an LLM.
        Returns a dict: {"response": ..., "goal_achieved": ...}
        """
        persona_data_str = json.dumps(self.persona.get(PersonaKeys.PERSONA_DATA, {}))
        persona_goal = self.persona.get(PersonaKeys.GOAL, "No goal specified.")

        try:
            system_prompt_template = self.persona.get(PersonaKeys.CLARIFICATION_PROMPT, "")
            system_prompt = system_prompt_template.format(
                goal=persona_goal,
                persona_data=persona_data_str
            )
            # Removed redundant JSON instruction, as it is now in _call_openai_api
        except (KeyError, TypeError) as e:
            logger.info(f"   [TestAgent ERROR] Invalid '{PersonaKeys.CLARIFICATION_PROMPT}' template: {e}")
            return {"response": f"Error: Invalid {PersonaKeys.CLARIFICATION_PROMPT} in persona.", "goal_achieved": False}

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

            Based on your instructions and the conversation so far, check whether all objectives in the goal are met or not.
            If not all objectives are met, as user's persona, provide the specific information requested by the agent.
            Your answer in JSON format with this exact structure: "
            '{{"goal_achieved": true/false, "information": "your answer as the user persona"}}'
        """).strip()

        response_text = await self._call_openai_api(system_prompt, user_prompt)

        if response_text.startswith("Error:"):
            return {"information": response_text, "goal_achieved": False}

        # Try to parse as JSON
        try:
            result = json.loads(response_text)
            if isinstance(result, dict) and "information" in result and "goal_achieved" in result:
                return result
        except Exception as e:
            logger.info(f"   [TestAgent ERROR] Could not parse LLM response as JSON: {e}. Raw: {response_text}")

        # Fallback: heuristic for goal_achieved
        goal_achieved = any(
            phrase in response_text.lower() for phrase in [
                "conversation is complete", "goal achieved", "no further action", "thank you", "done"
            ]
        )
        return {"information": response_text, "goal_achieved": goal_achieved}

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
        logger.info(f"Goal: {self.persona.get(PersonaKeys.GOAL, 'Complete booking')}")
        
        self.conversation_history = [] # Reset history for new test

        # The TestAgent generates the initial query
        current_query = await self._generate_initial_query()
        if current_query.startswith("Error:"):
            logger.info("   [TestAgent FATAL] Could not generate initial query.")
            return self.conversation_history # Fail fast if LLM fails

        logger.info(f"Initial Query (Generated): '{current_query}'")

        for turn in range(max_turns):
            # Bold for turn header
            logger.info(f"\n\033[1m--- Turn {turn + 1} ---\033[0m")
            # 1. TestAgent "speaks" to the AppAgent (its tool)
            # Cyan for TestAgent -> AppAgent
            logger.info(f"\033[96m[TestAgent -> AppAgent]: {current_query}\033[0m")
            self.conversation_history.append({'user': current_query})
            # 2. AppAgent processes the request (calling the generic tool)
            try:
                # current_query is now guaranteed to be a string
                app_response = await self.safe_tool_to_test(current_query)
            except Exception as e:
                logger.info("\n--- Test Failed ---")
                logger.info(f"The application tool itself raised an exception: {e}")
                return self.conversation_history
            # 3. TestAgent "hears" the AppAgent's response
            # Green for AppAgent -> TestAgent
            logger.info(f"\033[92m[AppAgent -> TestAgent]: {json.dumps(app_response, indent=2)}\033[0m")
            # Store the *dictionary* response for full context
            response_text = app_response.get("response")
            self.conversation_history.append({'assistant': response_text})

            # 4. TestAgent analyzes the response
            # Let the LLM decide if the conversation is complete or needs clarification
            simulated_answer = await self._simulate_user_answer_llm(response_text)
            #logger.info(f"[TestAgent (LLM) Simulated Answer]: {json.dumps(simulated_answer, indent=2)}")
            
            # Error check (string or None)
            if simulated_answer is None or (isinstance(simulated_answer, str) and simulated_answer.startswith("Error:")):
                logger.info("\n--- Test Ended ---")
                logger.info(f"TestAgent (LLM) did not generate a further answer. Final result: {response_text}")
                return self.conversation_history
            
            # --- FIX 3: Extract the string 'information' from the dictionary ---
            # This addresses the validation error: 'Input should be a valid string'
            next_user_message = simulated_answer.get("information")
            goal_achieved = simulated_answer.get("goal_achieved", False)

            # Heuristic: If the LLM says the conversation is complete, end
            if goal_achieved:
                logger.info("\n--- Test Succeeded ---")
                logger.info(f"Final Result: {response_text}")
                return self.conversation_history
                
            # Set the next query to be the *string* content
            current_query = next_user_message
            
            if current_query is None:
                # Fail if 'response' key was missing
                logger.info("\n--- Test Ended ---")
                logger.info("TestAgent (LLM) response was malformed (missing 'response' key).")
                return self.conversation_history

        logger.info(f"\n--- Test Ended: Max turns ({max_turns}) reached ---")
        return self.conversation_history