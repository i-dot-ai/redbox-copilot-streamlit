from redbox.llm.prompts.summary import (
    SUMMARY_KEY_ACTIONS_TASK_PROMPT,
    SUMMARY_KEY_DATES_TASK_PROMPT,
    SUMMARY_KEY_DISCUSSION_TASK_PROMPT,
    SUMMARY_KEY_PEOPLE_TASK_PROMPT,
    SUMMARY_TASK_PROMPT,
)
from redbox.models.summary import SummaryTask

# region ===== TASKS =====

summary_task = SummaryTask(
    id="summary",
    title="Summary",
    prompt_template=SUMMARY_TASK_PROMPT,
)
key_dates_task = SummaryTask(
    id="key_dates",
    title="Key Dates",
    prompt_template=SUMMARY_KEY_DATES_TASK_PROMPT,
)
key_actions_task = SummaryTask(
    id="key_actions",
    title="Key Actions",
    prompt_template=SUMMARY_KEY_ACTIONS_TASK_PROMPT,
)
key_people_task = SummaryTask(
    id="key_people",
    title="Key People",
    prompt_template=SUMMARY_KEY_PEOPLE_TASK_PROMPT,
)
key_discussion_task = SummaryTask(
    id="key_discussion",
    title="Key Discussion",
    prompt_template=SUMMARY_KEY_DISCUSSION_TASK_PROMPT,
)
# endregion
