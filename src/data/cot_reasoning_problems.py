"""
Chain-of-Thought Reasoning Problems for Disambiguation Experiments.

Hand-crafted problems requiring deliberation between two answer options.
Each problem has reference statements for computing H1/H2 centroids.
"""

from dataclasses import dataclass


@dataclass
class ReasoningProblem:
    """A reasoning problem with two competing answer options."""
    problem_id: str
    question: str
    h1_answer: str          # The H1 answer option
    h2_answer: str          # The H2 answer option
    correct: str            # "H1" or "H2"
    h1_reference_statements: list[str]  # Unambiguous statements asserting H1
    h2_reference_statements: list[str]  # Unambiguous statements asserting H2


REASONING_PROBLEMS = [
    ReasoningProblem(
        problem_id="prime_97",
        question="Is 97 a prime number? Think step by step, considering whether any numbers divide it evenly, before giving your final answer.",
        h1_answer="yes",
        h2_answer="no",
        correct="H1",
        h1_reference_statements=[
            "Yes, 97 is definitely a prime number. It is not divisible by any integer other than 1 and itself.",
            "97 is prime. I checked all possible factors up to its square root and none divide it evenly.",
            "The number 97 is indeed prime, as confirmed by checking divisibility by 2, 3, 5, and 7.",
        ],
        h2_reference_statements=[
            "No, 97 is not a prime number. It can be divided evenly by other integers besides 1 and itself.",
            "97 is composite, not prime. It has factors other than 1 and 97.",
            "The number 97 is not prime because it is divisible by smaller numbers.",
        ],
    ),
    ReasoningProblem(
        problem_id="bat_ball",
        question="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think carefully step by step before answering.",
        h1_answer="$0.05",
        h2_answer="$0.10",
        correct="H1",
        h1_reference_statements=[
            "The ball costs $0.05. If the ball is $0.05 then the bat is $1.05, and together they cost $1.10.",
            "The answer is 5 cents. Setting up the equation: ball + (ball + 1.00) = 1.10, so 2*ball = 0.10, ball = 0.05.",
            "The ball costs five cents, not ten cents. The intuitive answer of $0.10 is wrong because then the bat would cost $1.10.",
        ],
        h2_reference_statements=[
            "The ball costs $0.10. The bat costs $1.00 and the ball costs $0.10, totaling $1.10.",
            "The answer is 10 cents. If the total is $1.10 and the bat is $1.00, the ball must be $0.10.",
            "The ball costs ten cents. Subtracting the bat's cost of $1.00 from the total of $1.10 gives $0.10.",
        ],
    ),
    ReasoningProblem(
        problem_id="tomato",
        question="Is a tomato a fruit or a vegetable? Think step by step about both the botanical and culinary definitions before giving your answer.",
        h1_answer="fruit",
        h2_answer="vegetable",
        correct="H1",
        h1_reference_statements=[
            "A tomato is a fruit. Botanically, it develops from the flower of a plant and contains seeds.",
            "Tomatoes are fruits. By scientific definition, any seed-bearing structure that develops from a flower is a fruit.",
            "The tomato is classified as a fruit in biology because it is the mature ovary of a flowering plant.",
        ],
        h2_reference_statements=[
            "A tomato is a vegetable. In culinary practice and common usage, tomatoes are treated as vegetables.",
            "Tomatoes are vegetables. They are used in savory dishes and legally classified as vegetables.",
            "The tomato is a vegetable in the kitchen and in commerce, regardless of its botanical classification.",
        ],
    ),
    ReasoningProblem(
        problem_id="syllogism",
        question="All roses are flowers. Some flowers fade quickly. Does it follow that all roses fade quickly? Think step by step about the logical structure before answering.",
        h1_answer="no",
        h2_answer="yes",
        correct="H1",
        h1_reference_statements=[
            "No, it does not follow. Just because some flowers fade quickly does not mean all flowers do, so roses may or may not fade quickly.",
            "The conclusion does not follow logically. 'Some flowers fade quickly' does not imply 'all flowers fade quickly,' so we cannot conclude anything about all roses.",
            "No. This is a logical fallacy. The premise only says SOME flowers fade quickly, not all, so we cannot conclude that all roses fade quickly.",
        ],
        h2_reference_statements=[
            "Yes, all roses fade quickly. Since roses are flowers and flowers fade quickly, roses must also fade quickly.",
            "Yes, it follows logically. Roses are flowers, and flowers fade quickly, therefore roses fade quickly.",
            "The conclusion follows: all roses are flowers, flowers fade quickly, so all roses fade quickly.",
        ],
    ),
    ReasoningProblem(
        problem_id="feathers_bricks",
        question="Which is heavier: a pound of feathers or a pound of bricks? Think step by step before giving your answer.",
        h1_answer="same weight",
        h2_answer="bricks",
        correct="H1",
        h1_reference_statements=[
            "They weigh the same. A pound is a pound regardless of the material. Both weigh exactly one pound.",
            "Neither is heavier. A pound of feathers and a pound of bricks both weigh one pound by definition.",
            "They are equal in weight. The question specifies one pound of each, so they are identical in mass.",
        ],
        h2_reference_statements=[
            "The bricks are heavier. Bricks are a denser, heavier material than feathers.",
            "A pound of bricks is heavier because bricks are much denser and heavier than feathers.",
            "Bricks weigh more. Common sense tells us that brick material is heavier than feather material.",
        ],
    ),
    ReasoningProblem(
        problem_id="monty_hall",
        question="In the Monty Hall problem: you pick door 1, the host opens door 3 showing a goat. Should you switch to door 2 or stay with door 1? Think step by step about the probabilities.",
        h1_answer="switch",
        h2_answer="stay",
        correct="H1",
        h1_reference_statements=[
            "You should switch doors. Switching gives you a 2/3 probability of winning, while staying gives only 1/3.",
            "Switch to door 2. The host's reveal concentrates the 2/3 probability onto the remaining door.",
            "Always switch. The math shows switching wins 2 out of 3 times on average.",
        ],
        h2_reference_statements=[
            "You should stay with door 1. After the host opens a door, it's now 50/50 between the remaining two doors.",
            "Stay with your original choice. There are two doors left, so each has an equal chance of hiding the prize.",
            "It doesn't matter, but stay with door 1. Your initial choice has a 50% chance now that one door is eliminated.",
        ],
    ),
    ReasoningProblem(
        problem_id="repeating_decimal",
        question="Is 0.999... (repeating) exactly equal to 1? Think step by step using mathematical reasoning before answering.",
        h1_answer="yes",
        h2_answer="no",
        correct="H1",
        h1_reference_statements=[
            "Yes, 0.999 repeating is exactly equal to 1. This can be proven by letting x = 0.999..., then 10x = 9.999..., so 9x = 9 and x = 1.",
            "0.999... equals 1 exactly. There is no number between 0.999... and 1, proving they are the same number.",
            "Yes, they are equal. 1/3 = 0.333..., so 3 times 1/3 = 3 times 0.333... = 0.999... = 1.",
        ],
        h2_reference_statements=[
            "No, 0.999 repeating is not equal to 1. It approaches 1 but never quite reaches it.",
            "0.999... is infinitely close to 1 but not exactly equal. There is always an infinitesimally small difference.",
            "They are not equal. 0.999... is a limit that approaches 1, but the two numbers are distinct.",
        ],
    ),
    ReasoningProblem(
        problem_id="three_gloves",
        question="You have a drawer with an unknown mix of left and right gloves. If you pull out 3 gloves, are you guaranteed to have a matching pair (two left or two right)? Think step by step.",
        h1_answer="yes",
        h2_answer="no",
        correct="H1",
        h1_reference_statements=[
            "Yes, you are guaranteed a matching pair. By the pigeonhole principle, with 3 gloves and 2 types, at least 2 must be the same type.",
            "You are guaranteed a pair. With only two categories (left and right), picking 3 ensures at least two match.",
            "Yes. Three gloves into two categories means at least one category has two or more gloves.",
        ],
        h2_reference_statements=[
            "No, you might not get a matching pair. You could pull out all different gloves without getting a matching pair.",
            "You are not guaranteed a pair. It depends on the mix of gloves in the drawer.",
            "No guarantee. If the drawer has very few of one type, you might get an unmatched set.",
        ],
    ),
]


def get_reasoning_problems() -> list[ReasoningProblem]:
    """Return all reasoning problems."""
    return REASONING_PROBLEMS.copy()


def get_problem_by_id(problem_id: str) -> ReasoningProblem:
    """Get a specific problem by ID."""
    for p in REASONING_PROBLEMS:
        if p.problem_id == problem_id:
            return p
    raise ValueError(f"Unknown problem: {problem_id}")
