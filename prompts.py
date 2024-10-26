# The point of this prompt is just to see if a model can solve the problem correctly.
# It's roughly meant to mimic the minimal prompting that we would assume a user would do.
# The boxing is an attempt to make the verifier's job easier.
STRAIGHT_SHOT_SOLUTION_PROMPT = """
{problem}

Please solve this problem step-by-step, boxing the final answer.
"""


# NOTE: You can use {{}} instead of {} in exemplars; this will become a {}
# TODO: Re-add an exemplar for the verification reasoning.
VERIFY_SOLUTION_PROMPT = """
Here's a ground-truth problem and its solution:
<problem>
{problem}
</problem>
<solution>
{solution}
</solution>

Here's a candidate solution that may or may not be correct:
<candidate_solution>
{candidate_solution}
</candidate_solution>

The candidate solution _should_ have boxed (e.g. using the \\boxed{{...}} command) the answers to both explicitly stated subproblems and the final answer.

Given the above information, reason about whether the candidate solution is correct, where correctness is defined as producing a correct final answer.

First, reason about whether the solution is correct in <verification_reasoning></verification_reasoning> tags.
    - To do this, first state the final answer of the ground truth solution detailed in <solution> tags above.
    - Then, state the final answer of the candidate solution detailed in the <candidate_solution> tags above.
    - Finally, reason about whether the candidate solution is correct, specifically indicating the step and manner in which the reasoning may have gone wrong, if it did.
    - If the correct answer is produced in the candidate solution but not correctly boxed, that should still be considered as a Correct solution.
Make sure to remember to close your <verification_reasoning> tag with a </verification_reasoning> tag.

Then, determine whether the candidate solution is either "Correct" or "Incorrect" in <verification_result></verification_result> tags, given your reasoning.
In terms of structure, a good verification result might look like:
<verification_result>
Incorrect
</verification_result>
or
<verification_result>
Correct
</verification_result>
Make sure to remember to close your <verification_result> tag with a </verification_result> tag.
"""

# Note: We basically assume that the reasoning DID go wrong; if it went right, we'll just ignore whatever gets generated for these.
# Think: Is "determine" the right verb?
VERIFY_SOLUTION_PROMPT_WITH_PREFIX = (
    VERIFY_SOLUTION_PROMPT
    + """

Next, inside <prefix_reasoning></prefix_reasoning> tags: Given the reasoning within the <verification_reasoning> tags, determine the specific point in the original candidate solution where the reasoning went wrong. Explain the way in which this reasoning went wrong. Restate in double-quotes (without changing the wording) the specific portion of the original candidate solution that makes this error.
In terms of structure, a good prefix reasoning might look like:
<prefix_reasoning>
The error occurs when the candidate solution states: "We can write $\sqrt{{13}}$ as the sum of a whole number and a fractional part: $\sqrt{{13}}=3+0.7$."
</prefix_reasoning>
Make sure to remember to close your <prefix_reasoning> tag with a </prefix_reasoning> tag.

Finally, inside <verification_prefix></verification_prefix> tags, explicitly restate the contents of the original candidate solution UP TO AND INCLUDING the specific point in the original candidate solution where the reasoning went wrong, as described in <prefix_reasoning>, resulting in a truncated version of the candidate solution.
    - You should include the "Step" prefixes for each step in the candidate solution, if they are present.
    - It is ABSOLUTELY CRITICAL that you do not modify the specific wording, structure, or intent of the candidate solution.
    - Do NOT include any of your own analysis of the candidate solution in the <verification_prefix> tags.
    - DO NOT include the <candidate_solution> tags as part of your verification prefix. Only consider the text between the <candidate_solution> tags to be the candidate solution.
    - Make sure to close your <verification_prefix> tag with a </verification_prefix> tag.
    - Reiterating to highlight importance, the content in these verification_prefix tags should INCLUDE the erroneous step that should have been identified in the <prefix_reasoning> section.
In terms of structure, a good verification prefix might look like:
<verification_prefix>
We are asked to find the integer and decimal parts of $\sqrt{{13}}+1$.

Step 1: We can write $\sqrt{{13}}$ as the sum of a whole number and a fractional part: $\sqrt{{13}}=3+0.7$.
</verification_prefix>
Note that the prefix ends in the step identified in the <prefix_reasoning> step, if it is identified.
Make sure to remember to close your <verification_prefix> tag with a </verification_prefix> tag.
"""
)

COMPLETION_PROMPT_USER = """
{problem}

Please solve this problem step-by-step, boxing the final answer.
"""

# Do we even want this preamble?
COMPLETION_PROMPT_ASSISTANT = """
Certainly. Here is the step-by-step reasoning and final answer to the problem:

{prefix}
"""

COMPLETION_TEMPLATE = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_turn}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{assistant_turn}"""
