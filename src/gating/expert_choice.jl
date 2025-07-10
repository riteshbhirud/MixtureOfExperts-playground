"""
    ExpertChoiceGating(capacity_factor::Float32)

Expert choice routing where experts select tokens.
Each expert selects top tokens up to capacity.
"""
struct ExpertChoiceGating <: GatingMechanism
    capacity_factor::Float32
end

function compute_gates(gate::ExpertChoiceGating, router_logits::AbstractMatrix)
    num_experts, batch_size = size(router_logits)
    capacity = ceil(Int, batch_size * gate.capacity_factor / num_experts)
    
    scores = router_logits'  
    
    expert_choice_indices = zeros(Int, capacity, num_experts)
    expert_choice_gates = zeros(Float32, capacity, num_experts)
    
    router_probs = softmax(router_logits; dims=1)
    
    for expert in 1:num_experts
        expert_scores = scores[:, expert]
        num_selections = min(capacity, batch_size)
        
        topk_tokens = partialsortperm(expert_scores, 1:num_selections, rev=true)
        
        for (idx, token) in enumerate(topk_tokens)
            expert_choice_indices[idx, expert] = token
            expert_choice_gates[idx, expert] = router_probs[expert, token]
        end
    end
    

    max_experts_per_token = capacity 
    expert_indices = zeros(Int, max_experts_per_token, batch_size)
    expert_gates = zeros(Float32, max_experts_per_token, batch_size)
    
    token_expert_count = zeros(Int, batch_size)
    
    for expert in 1:num_experts
        for slot in 1:capacity
            token = expert_choice_indices[slot, expert]
            if token > 0 && token <= batch_size
                token_expert_count[token] += 1
                if token_expert_count[token] <= max_experts_per_token
                    expert_indices[token_expert_count[token], token] = expert
                    expert_gates[token_expert_count[token], token] = expert_choice_gates[slot, expert]
                end
            end
        end
    end
    
    return expert_indices, expert_gates, router_probs
end