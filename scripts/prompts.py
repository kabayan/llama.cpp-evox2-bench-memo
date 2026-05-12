"""Real-world prompts used in the bench.

Three single-turn user prompts, no system message. The bench runs each prompt
through the chat-template path (`--jinja --reasoning-format auto` on llama-server),
so the actual token count seen by the model depends on the target model's template.
Approximate user-content token counts (Qwen3.5 tokenizer): 115 / 85 / 95.
"""

PROMPTS: list[tuple[str, str]] = [
    (
        "P_code",
        "Write a Python function `binary_search(arr: list[int], target: int) -> int` "
        "that returns the index of `target` in a sorted list `arr`, or -1 if not found. "
        "Include type hints, a complete docstring, and at the end add a small test block "
        "that demonstrates it on `[1, 3, 5, 7, 9, 11, 13, 15, 17]` searching for 7 and for 4, "
        "printing the results. Then briefly explain the time complexity in two sentences.",
    ),
    (
        "P_chat",
        "I'm planning a 3-day trip to Kyoto in mid-November. Please suggest a concrete "
        "day-by-day itinerary covering famous temples, local food spots, and one quiet area "
        "away from heavy tourist crowds. Include rough morning/afternoon/evening timings "
        "and short walking distances between stops. Keep it practical for a solo traveler.",
    ),
    (
        "P_reason",
        "Two trains start at the same morning. Train A leaves Tokyo for Osaka at 9:00 AM "
        "at a constant 250 km/h. Train B leaves Osaka for Tokyo at 9:30 AM at 200 km/h. "
        "The Tokyo-Osaka distance is 550 km. (1) At what time do they meet? "
        "(2) How far from Tokyo do they meet? Show the algebra step by step, "
        "then state the final answers clearly.",
    ),
]


def get(name: str) -> str:
    """Return the text of a named prompt, or raise KeyError."""
    for n, text in PROMPTS:
        if n == name:
            return text
    raise KeyError(f"unknown prompt {name!r}; expected one of {[n for n, _ in PROMPTS]}")
