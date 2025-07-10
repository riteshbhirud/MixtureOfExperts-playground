"""
    GatingMechanism

Abstract type for all gating mechanisms.
From Stanford CS336: Many routing algorithms boil down to "choose top-k"
"""
abstract type GatingMechanism end

"""
    LoadBalancingLoss

Abstract type for load balancing losses.
"""
abstract type LoadBalancingLoss end

"""
    compute_gates(gate::GatingMechanism, router_logits::AbstractMatrix)

Compute expert assignments and gating values.
Returns: (expert_indices, expert_gates, router_probs)
"""
function compute_gates end