from collections import defaultdict
from datetime import date
from typing import Optional

from langchain.schema import HumanMessage, SystemMessage

from redbox.llm.prompts.summary import SUMMARY_COMBINATION_TASK_PROMPT


class SummaryTag(object):
    """A class for combining Summary task outputs into a cohesive briefing.

    Args:
        object (_type_): _description_
    """

    def __init__(self, summarys: list[dict]) -> None:
        self.summarys = summarys
        self.combined_summary_tasks = defaultdict(list)

        for individual_summary_dict in self.summarys:
            for task_id in individual_summary_dict["task_outputs"]:
                self.combined_summary_tasks[task_id].append(individual_summary_dict["task_outputs"][task_id])
        self.combined_summary_dict: dict = {"combined_task_outputs": {}}

    def combine_summary_task_outputs(
        self,
        task_outputs: list[dict],
        task_id,
        user_info,
        llm,
        callbacks: Optional[list] = None,
    ):
        """Combine the outputs of a task across all summarys."""
        summary_payload = ""

        for i, individual_task_payload in enumerate(task_outputs):
            summary_payload += (
                f"Summary {i}: {individual_task_payload['title']}:\n{individual_task_payload['content']}\n\n\n"
            )

        messages_to_send = [
            SystemMessage(
                content=SUMMARY_COMBINATION_TASK_PROMPT.format(
                    current_date=date.today().isoformat(),
                    user_info=user_info,
                )
            ),
            HumanMessage(content=summary_payload),
        ]

        result = llm(messages_to_send, callbacks=callbacks or [])
        self.combined_summary_dict[task_id] = result
        return result
