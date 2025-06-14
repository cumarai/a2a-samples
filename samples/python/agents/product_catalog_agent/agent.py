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


def search_products(
    product_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Search for products in the catalog by matching product_name in descriptions.

    Args:
        product_name (str): The name of the product to search for.

    Returns:
        dict[str, Any]: A dictionary containing the request_id, list of found products, and a message.
    """
    request_id = 'request_id_' + str(random.randint(1000000, 9999999))
    request_ids.add(request_id)
    
    # Path to the products CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'products.csv')
    
    # Load the products from CSV into a DataFrame
    try:
        df = pd.read_csv(csv_path)
        
        # If no product name provided, return all products
        if not product_name:
            products_list = df.to_dict('records')
            product_names = [product['name'] for product in products_list]
            message = f"Encontramos {len(products_list)} productos en nuestro catálogo: {', '.join(product_names)}"
            return {
                'request_id': request_id,
                'products_list': products_list,
                'message': message
            }
        
        # Convert product_name and descriptions to lowercase for case-insensitive search
        search_term = product_name.lower()
        df['description_lower'] = df['description'].str.lower()
        
        # Filter products where the search term is in the description
        filtered_df = df[df['description_lower'].str.contains(search_term)]
        
        # Convert filtered results to list of dictionaries
        products_list = filtered_df.drop('description_lower', axis=1).to_dict('records')
        
        # Create message based on search results
        if products_list:
            product_names = [product['name'] for product in products_list]
            message = f"Encontramos {len(products_list)} productos que coinciden con '{product_name}': {', '.join(product_names)}"
        else:
            message = f"No encontramos ningún producto que coincida con '{product_name}' en nuestro catálogo."
        
        return {
            'request_id': request_id,
            'products_list': products_list,
            'message': message
        }
        
    except Exception as e:
        # Handle errors (file not found, parsing errors, etc.)
        return {
            'request_id': request_id,
            'products_list': [],
            'message': f"Error al buscar productos: {str(e)}"
        }


class ProductCatalogAgent:
    """An agent that handles product catalog requests."""

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
        return 'Processing the product catalog request...'

    def _build_agent(self) -> LlmAgent:
        """Builds the LLM agent for the product catalog agent."""
        return LlmAgent(
            model='gemini-2.0-flash-001',
            name='product_catalog_agent',
            description=(
                'This agent handles the product catalog search process for the customer'
                ' given the name of the product.'
            ),
            instruction="""
    You are an agent who handles the product catalog search process for customers.

    When asked about a product:
    1. Use the search_products tool to search for products in our catalog
    2. Return the exact message provided in the 'message' field of the tool's response
    3. If products are found, you can provide additional details about them if requested
    4. Be polite and professional in your interactions

    The search_products tool searches in our product catalog database and returns matching products based on the description.
    """,
            tools=[
                search_products,
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
