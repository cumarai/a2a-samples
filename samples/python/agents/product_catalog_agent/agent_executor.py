import json


from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_parts_message,
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from agent import ProductCatalogAgent


class ProductCatalogAgentExecutor(AgentExecutor):
    """Product Catalog AgentExecutor Example."""

    def __init__(self):
        self.agent = ProductCatalogAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        # This agent always produces Task objects. If this request does
        # not have current task, create a new one and use it.
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)
        # invoke the underlying agent, using streaming results. The streams
        # now are update events.
        async for item in self.agent.stream(query, task.contextId):
            is_task_complete = item['is_task_complete']
            artifacts = None
            if not is_task_complete:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        item['updates'], task.contextId, task.id
                    ),
                )
                continue
            # If the response is a dictionary, handle our product catalog response format
            if isinstance(item['content'], dict):
                # Check if it's our product catalog response format
                if 'message' in item['content']:
                    # Extract the message field and use it as text response
                    message = item['content']['message']
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(
                            message, task.contextId, task.id
                        ),
                        final=True,
                    )
                    
                    # If there are products, add them as a data artifact
                    if 'products_list' in item['content'] and item['content']['products_list']:
                        products_data = {
                            'products': item['content']['products_list']
                        }
                        await updater.add_artifact(
                            [Part(root=DataPart(data=products_data))], 
                            name='products'
                        )
                    continue
                else:
                    # Si el diccionario no tiene el formato esperado, tratarlo como un error
                    await updater.update_status(
                        TaskState.failed,
                        new_agent_text_message(
                            f'Formato de respuesta inesperado: {item["content"]}',
                            task.contextId,
                            task.id,
                        ),
                        final=True,
                    )
                    break
            else:
                # Emit the appropriate events
                await updater.add_artifact(
                    [Part(root=TextPart(text=item['content']))], name='form'
                )
                await updater.complete()
                break

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
