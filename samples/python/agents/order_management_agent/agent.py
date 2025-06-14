import json
import random
import os
import pandas as pd
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


def search_orders(
    order_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Search for orders in the system.

    Args:
        order_id (str): The ID of the order to search for.

    Returns:
        dict[str, Any]: A dictionary containing the request_id, list of found orders, and a message.
    """
    request_id = 'request_id_' + str(random.randint(1000000, 9999999))
    request_ids.add(request_id)
    
    # Path to the orders CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'orders.csv')
    
    # Load the orders from CSV into a DataFrame
    try:
        df = pd.read_csv(csv_path)
        
        # If no order id provided, return all orders
        if not order_id:
            orders_list = df.to_dict('records')
            order_ids = [str(order['id']) for order in orders_list]
            message = f"Encontramos {len(orders_list)} órdenes en nuestro sistema: {', '.join(order_ids)}"
            return {
                'request_id': request_id,
                'orders_list': orders_list,
                'message': message
            }
        
        # Filter orders by ID (convert both to string for comparison)
        filtered_df = df[df['id'].astype(str) == order_id]
        
        # Convert filtered results to list of dictionaries
        orders_list = filtered_df.to_dict('records')
        
        # Create message based on search results
        if orders_list:
            message = f"Encontramos la orden con ID '{order_id}' en nuestro sistema."
        else:
            message = f"No encontramos ninguna orden con ID '{order_id}' en nuestro sistema."
        
        return {
            'request_id': request_id,
            'orders_list': orders_list,
            'message': message
        }
        
    except Exception as e:
        # Handle errors (file not found, parsing errors, etc.)
        return {
            'request_id': request_id,
            'orders_list': [],
            'message': f"Error al buscar órdenes: {str(e)}"
        }


class OrderManagementAgent:
    """An agent that handles order management requests."""

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
        return 'Processing the order management request...'

    def _build_agent(self) -> LlmAgent:
        """Builds the LLM agent for the order management agent."""
        return LlmAgent(
            model='gemini-2.0-flash-001',
            name='order_management_agent',
            description=(
                'This agent handles the order management process for the customer'
                ' given the order id.'
            ),
            instruction="""
    You are an agent who handles the order management process for customers.

    When asked about an order:
    1. Use the search_orders tool to search for orders in our catalog
    2. Return the exact message provided in the 'message' field of the tool's response
    3. If orders are found, you can provide additional details about them if requested
    4. Be polite and professional in your interactions

    The search_orders tool searches in our order catalog database and returns matching orders based on the description.
    """,
            tools=[
                search_orders,
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
