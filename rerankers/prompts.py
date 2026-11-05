# Dataset prompts from the original Rank1 paper and implementation:
# https://arxiv.org/abs/2502.18418
# https://github.com/orionw/rank1

PROMPT_DICT = {
    "aops": """Find different but similar math problems to FILL_QUERY_HERE\n\nA document is relevant if it uses the same class of functions and shares **any** overlapping techniques.""",

    "theoremqa_questions": """Find a passage which uses the same mathematical process as this one: FILL_QUERY_HERE""",

    "leetcode": """I am looking to find different problems that share similar data structures (of any kind) or algorithms (e.g. DFS, DP, sorting, traversals, etc.). I am looking for problems that share one or both of these similarities to this:
    
FILL_QUERY_HERE

Does this passage share any similarities? e.g. if there was a textbook on leetcode problems, this would be in the same book even though it could be in a different chapter. 


""",

    "pony": """I will use the programming language pony. Problem: FILL_QUERY_HERE

But to solve the problem above, I need to know things about pony. A passage is relevant if it contains docs that match **any** part (even basic parts) of the code I will have to write for the above program.""",

    "BrightRetrieval": """Can you find background information about the concepts used to answer the question:

FILL_QUERY_HERE

A passage is relevant if it contains background information about a **sub-concept** that someone might cite/link to when answering the above question."""

}

PROMPT_DICT["theoremqa_theorems"] = PROMPT_DICT["theoremqa_questions"]
PROMPT_DICT["biology"] = PROMPT_DICT["BrightRetrieval"]
PROMPT_DICT["earth_science"] = PROMPT_DICT["BrightRetrieval"]
PROMPT_DICT["economics"] = PROMPT_DICT["BrightRetrieval"]
PROMPT_DICT["psychology"] = PROMPT_DICT["BrightRetrieval"]
PROMPT_DICT["robotics"] = PROMPT_DICT["BrightRetrieval"]
PROMPT_DICT["stackoverflow"] = PROMPT_DICT["BrightRetrieval"]
PROMPT_DICT["sustainable_living"] = PROMPT_DICT["BrightRetrieval"]

def get_prompt(corpus_name):
    if corpus_name in PROMPT_DICT:
        return PROMPT_DICT[corpus_name]
    else:
        return None
