import dspy
import pandas as pd
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPRO

from app.setup import (
    METRIC_LM,
    MODEL_PATH,
    PROMPT_LM,
    STUDENT_LM,
    BasicScopeGenerator,
    ScopeGenerator,
)

# Optimizer params - https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/qa/hotpot/hotpotqa_with_MIPRO.ipynb
NUM_THREADS = 16
NUM_CANDIDATES = 10
INIT_TEMPERATURE = 1.0
NUM_TRIALS = 20
MAX_BOOTSTRAPPED_DEMOS = 1
MAX_LABELED_DEMOS = 2


# Dataset params
EXAMPLES_PATH = "data/examples.csv"
TRAIN_TEST_SPLIT = 0.9  # Randomly chosen


# Load dataset
df = pd.read_csv(EXAMPLES_PATH)
dataset = []
for title in df["title"]:
    dataset.append(
        dspy.Example(
            title=title,
        ).with_inputs("title")
    )
trainset, devset = dataset[: int(TRAIN_TEST_SPLIT * len(dataset))], dataset[int(TRAIN_TEST_SPLIT * len(dataset)) :]


# Define metric
class Assess(dspy.Signature):
    """Assess the quality of a task scope along the specified dimension."""

    title = dspy.InputField()
    description = dspy.InputField()
    acceptance_criteria = dspy.InputField()
    sub_tasks = dspy.InputField()
    assumptions = dspy.InputField()
    dependencies = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


def metric(gold, pred, trace=None):
    title, description, acceptance_criteria, sub_tasks, assumptions, dependencies = (
        gold.title,
        pred.description,
        pred.acceptance_criteria,
        pred.sub_tasks,
        pred.assumptions,
        pred.dependencies,
    )

    # Framing the base question for the assessment
    base_question = f"""
    The task title is: {title}.
    Based on this title, the provided task scope is:

    Description: {description}

    Acceptance Criteria:
    {acceptance_criteria}

    Sub-Tasks:
    {sub_tasks}

    Assumptions:
    {assumptions}

    Dependencies:
    {dependencies}
    """

    # Defining the assessment questions
    questions = [
        "Is the task description sufficiently detailed to provide a clear understanding of the expected outcomes?",
        "Do the acceptance criteria outline specific, measurable, and testable conditions for task completion?",
        "Are the sub-tasks logically broken down, ensuring each is actionable and contributes directly to the task's completion?",
        "Are all assumptions explicitly stated, realistic, and validated to prevent potential misunderstandings?",
        "Are dependencies clearly identified, including both internal and external factors that could impact task execution?",
        "Does the task scope consider potential risks and include mitigation strategies or contingency plans?",
        "Is the task scope aligned with project goals and constraints, including timelines, resources, and technological capabilities?",
        "Does the task include considerations for integration and compatibility with existing systems or workflows?",
    ]
    questions = [base_question + "\n" + question for question in questions]

    # Collect responses to the questions
    responses = []
    with dspy.context(lm=METRIC_LM):
        for question in questions:
            response = dspy.Predict(Assess)(
                title=title,
                description=description,
                acceptance_criteria=acceptance_criteria,
                sub_tasks=sub_tasks,
                assumptions=assumptions,
                dependencies=dependencies,
                assessment_question=question,
            )
            responses.append(response)

    # Convert the responses to boolean values and calculate the score
    results = [response.assessment_answer.lower() == "yes" for response in responses]
    score = sum(results)

    # Final evaluation logic
    if trace is not None:  # Provide detailed feedback if trace is enabled
        return score >= len(questions)  # True if all questions are affirmed
    return score / len(questions)  # Otherwise, return the proportion of positive responses as the score


# Optimization
def main():
    optimizer = MIPRO(
        metric=metric,
        prompt_model=PROMPT_LM,
        task_model=STUDENT_LM,
        num_candidates=NUM_CANDIDATES,
        init_temperature=INIT_TEMPERATURE,
        verbose=True,
    )
    compiled_program = optimizer.compile(
        student=ScopeGenerator(),
        trainset=trainset,
        num_trials=NUM_TRIALS,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
        eval_kwargs={"num_threads": NUM_THREADS, "display_progress": True, "display_table": 0},
    )
    compiled_program.save(MODEL_PATH)

    evaluate = Evaluate(devset=devset, metric=metric, num_threads=NUM_THREADS, display_progress=True)
    print(f"Vanilla LLM score: {evaluate(BasicScopeGenerator())}")
    print(f"Baseline program score: {evaluate(ScopeGenerator())}")
    print(f"Compiled program score: {evaluate(compiled_program)}")


if __name__ == "__main__":
    main()
