"""
    RandomGating(k::Int = 1)

Random expert selection for initial testing.
As requested: "at the beginning, just choose random expert"
"""
struct RandomGating <: GatingMechanism
    k::Int
end

RandomGating() = RandomGating(1)

function compute_gates(gate::RandomGating, router_logits::AbstractMatrix)
    num_experts, batch_size = size(router_logits)
    
    expert_indices = zeros(Int, gate.k, batch_size)
    expert_gates = ones(Float32, gate.k, batch_size) ./ gate.k
    
    for i in 1:batch_size
        selected = StatsBase.sample(1:num_experts, gate.k, replace=false)
        expert_indices[:, i] = selected
    end
    
    router_probs = ones(Float32, num_experts, batch_size) ./ num_experts
    
    return expert_indices, expert_gates, router_probs
end

"""
    NoBalancingLoss

No load balancing for initial testing.
"""
struct NoBalancingLoss <: LoadBalancingLoss end

compute_loss(::NoBalancingLoss, args...) = 0.0f0