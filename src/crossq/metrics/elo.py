


def compute_expected_scores(rating_a: float, rating_b: float) -> tuple[float, float]:

    e_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    e_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

    return e_a, e_b


def update_elo_ratings(old_rating_a: float, 
                       old_rating_b: float, 
                       k_factor: int,
                       result: int) -> tuple[float, float]:
    # results is the outcome of the game:
    # 1 - agent a has won
    # 0 - draw
    # -1 - agent b has won
    e_a, e_b = compute_expected_scores(old_rating_a, old_rating_b)  # first compute the  expected scores of the agents

    score_a, score_b = int(result == 1), int(result == -1)

    elo_a = old_rating_a + k_factor * (score_a - e_a)
    elo_b = old_rating_b + k_factor * (score_b - e_b)

    return elo_a, elo_b