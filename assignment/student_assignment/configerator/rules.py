"""Author: @hguru.

This file contains all the rules that a config must follow.
A config can be legal syntactically but not semantically; this ruleset
encodes illegal combinations of constraints in more detail, to avoid errors
"""

import rules


@rules.predicate
def is_using_utility_model(config):
    return config["utility-model"]["enable"]


@rules.predicate
def is_using_sibling_access(config):
    return config["sibling-access"]


ruleset = [~is_using_utility_model | ~is_using_sibling_access]
