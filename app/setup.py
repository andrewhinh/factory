import os
import traceback

import dspy
from dotenv import load_dotenv
from dsp.utils import deduplicate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LINEAR_API_TOKEN = os.getenv("LINEAR_API_TOKEN")
LINEAR_WH_TOKEN = os.getenv("LINEAR_WH_TOKEN")


# LM
METRIC_MODEL = "gpt-3.5-turbo"
PROMPT_MODEL = "gpt-4-turbo-preview"
STUDENT_MODEL = "gpt-3.5-turbo"
MODEL_TYPE = "chat"
MAX_HOPS = (
    2  # for multi-hop reasoning - https://dspy-docs.vercel.app/docs/tutorials/simplified-baleen#building-the-pipeline
)

METRIC_LM = dspy.OpenAI(model=METRIC_MODEL, api_key=OPENAI_API_KEY, model_type=MODEL_TYPE)
PROMPT_LM = dspy.OpenAI(model=PROMPT_MODEL, api_key=OPENAI_API_KEY, model_type=MODEL_TYPE)
STUDENT_LM = dspy.OpenAI(model=STUDENT_MODEL, api_key=OPENAI_API_KEY, model_type=MODEL_TYPE)


# Problem
class GenerateScope(dspy.Signature):
    """
    Given a minimally described task (with just a title), expand it into a well-scoped ticket suitable for a software engineer to implement.
    The completed scope for the task must include descriptions, acceptance criteria, sub-tasks, assumptions, and dependencies in markdown format.
    Consider what details are necessary to turn a simple title into a fully scoped task.
    Think about industry best practices and what information engineers need to avoid ambiguity and back-and-forth.
    What information could be helpful to understand how to scope a ticket properly?

    e.g. "Create a login page" ->
    **Description:** Create a login page that allows users to sign in with their email and password.

    **Acceptance Criteria:**
    1. Users can enter their email and password.
    2. Users can click a "Sign In" button to log in.
    3. Users see an error message if they enter an incorrect email or password.

    **Sub-tasks:**
    1. Design the login page UI.
    2. Implement the login page frontend.
    3. Implement the login page backend.

    **Assumptions:**
    1. The backend API for user authentication is already implemented.
    2. The design for the login page is already approved.

    **Dependencies:**
    1. The backend API for user authentication.
    2. The approved design for the login page.
    """

    context = dspy.InputField(desc="may contain relevant context")
    title = dspy.InputField(desc="title of the task")
    description = dspy.OutputField(desc="description of the task")
    acceptance_criteria = dspy.OutputField(desc="acceptance criteria for the task")
    sub_tasks = dspy.OutputField(desc="sub-tasks for the task")
    assumptions = dspy.OutputField(desc="assumptions for the task")
    dependencies = dspy.OutputField(desc="dependencies for the task")


# Programs
class BasicScopeGenerator(dspy.Module):
    """Generates task scopes from titles."""

    def __init__(
        self,
        lm=STUDENT_LM,
    ):
        super().__init__()

        self.lm = lm
        self.generate_scope = dspy.Predict(GenerateScope)

    def forward(self, title):
        try:
            with dspy.context(lm=self.lm):
                pred = self.generate_scope(
                    context=[],
                    title=title,
                )
                description, acceptance_criteria, sub_tasks, assumptions, dependencies = (
                    pred.description,
                    pred.acceptance_criteria,
                    pred.sub_tasks,
                    pred.assumptions,
                    pred.dependencies,
                )
            return dspy.Prediction(
                description=description,
                acceptance_criteria=acceptance_criteria,
                sub_tasks=sub_tasks,
                assumptions=assumptions,
                dependencies=dependencies,
            )
        except Exception:
            return dspy.Prediction(scope="")


class ScopeGenerator(dspy.Module):
    """Generates task scopes from titles."""

    def __init__(
        self,
        lm=STUDENT_LM,
        max_hops=MAX_HOPS,
    ):
        super().__init__()

        self.lm = lm
        self.max_hops = max_hops
        self.generate_scope = [dspy.ChainOfThought(GenerateScope) for _ in range(max_hops)]

    def forward(self, title):
        context, description, acceptance_criteria, sub_tasks, assumptions, dependencies = [], "", "", "", "", ""

        for hop in range(self.max_hops):
            try:
                with dspy.context(lm=self.lm):
                    pred = self.generate_scope[hop](
                        context=context,
                        title=title,
                    )
                    description, acceptance_criteria, sub_tasks, assumptions, dependencies = (
                        pred.description,
                        pred.acceptance_criteria,
                        pred.sub_tasks,
                        pred.assumptions,
                        pred.dependencies,
                    )
                passages = [f"{description}\n{acceptance_criteria}\n{sub_tasks}\n{assumptions}\n{dependencies}"]
                context = deduplicate(context + passages)
            except Exception:
                passages = [traceback.format_exc()]
                context = deduplicate(context + passages)
        return dspy.Prediction(
            description=description,
            acceptance_criteria=acceptance_criteria,
            sub_tasks=sub_tasks,
            assumptions=assumptions,
            dependencies=dependencies,
        )


MODEL_PATH = "models/scope_generator.json"
