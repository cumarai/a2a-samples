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
from agent import FAQsManagementAgent


class FAQsManagementAgentExecutor(AgentExecutor):
    """FAQs Management AgentExecutor Example."""

    def __init__(self):
        self.agent = FAQsManagementAgent()

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
            # If the response is a dictionary, handle our FAQs response format
            if isinstance(item['content'], dict):
                # Check if it's our FAQs response format
                if 'message' in item['content']:
                    # Extract the message field and use it as text response
                    message = item['content']['message']
                    
                    # If we have FAQs content and a query, process it
                    if 'faqs_content' in item['content'] and 'query' in item['content']:
                        # Here we would normally process with the LLM
                        # For now, we'll just return the message
                        await updater.update_status(
                            TaskState.completed,
                            new_agent_text_message(
                                message, task.contextId, task.id
                            ),
                            final=True,
                        )
                    else:
                        # Just return the message if no FAQs content or query
                        await updater.update_status(
                            TaskState.completed,
                            new_agent_text_message(
                                message, task.contextId, task.id
                            ),
                            final=True,
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
