"""
    AuxiliaryFreeLoss(num_experts::Int; kwargs...)

DeepSeek V3 innovation: Auxiliary loss-free balancing using per-expert bias.

g'_{i,t} = {s_{i,t}, if s_{i,t} + b_i ∈ TopK({s_{j,t} + b_j | 1 ≤ j ≤ N_r}, K_r)
          {0, otherwise

The bias is updated online to encourage uniform distribution.
"""
mutable struct AuxiliaryFreeLoss <: LoadBalancingLoss
    bias::Vector{Float32}
    learning_rate::Float32
    momentum::Float32
    bias_momentum::Vector{Float32}
    update_freq::Int
    update_counter::Ref{Int}
    use_complementary_loss::Bool
    complementary_weight::Float32
end

function AuxiliaryFreeLoss(num_experts::Int; 
                          bias_init::Float32 = 0.0f0,
                          learning_rate::Float32 = 0.01f0,
                          momentum::Float32 = 0.9f0,
                          update_freq::Int = 1,
                          use_complementary_loss::Bool = true,
                          complementary_weight::Float32 = 0.001f0)
    bias = fill(bias_init, num_experts)
    bias_momentum = zeros(Float32, num_experts)
    return AuxiliaryFreeLoss(bias, learning_rate, momentum, bias_momentum, 
                            update_freq, Ref(0), use_complementary_loss, 
                            complementary_weight)
end

function update_bias!(loss_fn::AuxiliaryFreeLoss, expert_indices::AbstractMatrix)
    loss_fn.update_counter[] += 1
    
    if loss_fn.update_counter[] % loss_fn.update_freq != 0
        return
    end
    
    num_experts = length(loss_fn.bias)
    
    expert_counts = zeros(Float32, num_experts)
    total_assignments = 0
    
    for idx in expert_indices
        if idx > 0
            expert_counts[idx] += 1
            total_assignments += 1
        end
    end
    
    if total_assignments == 0
        return
    end
    
    target_count = total_assignments / num_experts
    
    for i in 1:num_experts
        gradient = (expert_counts[i] - target_count) / total_assignments
        
        loss_fn.bias_momentum[i] = loss_fn.momentum * loss_fn.bias_momentum[i] + 
                                  (1 - loss_fn.momentum) * gradient
        
        loss_fn.bias[i] -= loss_fn.learning_rate * loss_fn.bias_momentum[i]
    end
    
    loss_fn.bias .-= mean(loss_fn.bias)
end

function compute_loss(loss_fn::AuxiliaryFreeLoss, expert_indices::AbstractMatrix, 
                     router_probs::AbstractMatrix)
    update_bias!(loss_fn, expert_indices)
    
    if loss_fn.use_complementary_loss
        return loss_fn.complementary_weight * 
               compute_loss(SwitchTransformerLoss(1.0f0), expert_indices, router_probs)
    else
        return 0.0f0
    end
end

"""
    get_biased_logits(loss_fn::AuxiliaryFreeLoss, router_logits::AbstractMatrix)

Apply learned bias to router logits for auxiliary-free balancing.
"""
function get_biased_logits(loss_fn::AuxiliaryFreeLoss, router_logits::AbstractMatrix)
    return router_logits .+ reshape(loss_fn.bias, :, 1)
end