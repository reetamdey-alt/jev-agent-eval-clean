"""Deterministic synthetic case generators (v2 spec section 32).

Required generator families (all implemented in this package):

    minimal_pair
    counterfactual
    contradiction
    near_neighbor
    composition
    boundary
    long_context
    position_bias
    adversarial
    temporal_state
    recovery
    uncertainty

Every generator emits provenance metadata (generator, generator_version,
seed, parent_case_id, relation) and generated variants are marked
is_variant=True so they never enter the primary sample count.
"""
