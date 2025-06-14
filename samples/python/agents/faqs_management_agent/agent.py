import json
import random
import os
from typing import Any, AsyncIterable, List, Optional
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types


# Local cache of created request_ids for demo purposes.
request_ids = set()


def search_faqs(
    query: str,
) -> dict[str, Any]:
    """
    Search for answers to FAQs based on the query using the LLM.

    Args:
        query (str): The question from the user.

    Returns:
        dict[str, Any]: A dictionary containing the request_id, faqs_content, query, and a message.
    """
    request_id = 'request_id_' + str(random.randint(1000000, 9999999))
    request_ids.add(request_id)
    
    # Path to the FAQs text file
    faqs_path = os.path.join(os.path.dirname(__file__), 'data', 'faqs.txt')
    
    try:
        # Load the entire FAQs content
        with open(faqs_path, 'r', encoding='utf-8') as file:
            faqs_content = file.read()
        
        # The content will be included in the context for the LLM
        # The LLM will have access to all the information to answer the query
        return {
            'request_id': request_id,
            'faqs_content': faqs_content,
            'query': query,
            'message': "He encontrado información relevante para tu pregunta."
        }
        
    except Exception as e:
        return {
            'request_id': request_id,
            'faqs_content': '',
            'query': query,
            'message': f"Error al buscar información: {str(e)}"
        }


class FAQsManagementAgent:
    """An agent that handles general questions about the company, including FAQs, policies, schedules, and other information."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = 'remote_agent'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def get_processing_message(self) -> str:
        return 'Procesando tu consulta general sobre la empresa...'

    def _build_agent(self) -> LlmAgent:
        """Builds the LLM agent for the FAQs management agent."""
        return LlmAgent(
            model='gemini-2.0-flash-001',
            name='faqs_management_agent',
            description=(
                'This agent answers general questions about the company including policies,'
                ' schedules, locations, and any other information not related to specific product or order searches'
            ),
            instruction="""
    You are an agent who answers general questions about the company.

    When asked a question:
    1. Use the search_faqs tool to get the FAQs content
    2. Read through the FAQs content to find relevant information
    3. Provide a clear, concise answer based on the information in the FAQs
    4. If the exact question isn't covered in the FAQs, provide the most relevant information available
    5. Be polite and professional in your interactions
    6. Remember that you handle all general questions that are NOT about specific product searches or order lookups
    7. Your knowledge includes company policies, schedules, locations, contact information, and general company information

    The search_faqs tool provides you with the complete FAQs document containing all general company information.
    """,
            tools=[
                search_faqs,
            ],
        )

    async def stream(self, query, session_id) -> AsyncIterable[dict[str, Any]]:
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(
            role='user', parts=[types.Part.from_text(text=query)]
        )
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ''
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = '\n'.join(
                        [p.text for p in event.content.parts if p.text]
                    )
                elif (
                    event.content
                    and event.content.parts
                    and any(
                        [
                            True
                            for p in event.content.parts
                            if p.function_response
                        ]
                    )
                ):
                    response = next(
                        p.function_response.model_dump()
                        for p in event.content.parts
                    )
                yield {
                    'is_task_complete': True,
                    'content': response,
                }
            else:
                yield {
                    'is_task_complete': False,
                    'updates': self.get_processing_message(),
                }
