"""
    SwitchGating()

Switch Transformer gating - each token to exactly one expert.
Special case of TopK with k=1.
"""
struct SwitchGating <: GatingMechanism end

function compute_gates(gate::SwitchGating, router_logits::AbstractMatrix)
    return compute_gates(TopKGating(1), router_logits)
end

"""
    JitterGating(ε::Float32)

Switch Transformer uniform jitter (Fedus et al 2022):
if is_training:
    router_logits += random_uniform(minval=1-eps, maxval=1+eps)
"""
struct JitterGating <: GatingMechanism
    base_gate::GatingMechanism
    ε::Float32
end

function compute_gates(gate::JitterGating, router_logits::AbstractMatrix; 
                      training::Bool = false)
    if training
        jitter = 1.0f0 .+ (2.0f0 * rand(Float32, size(router_logits)) .- 1.0f0) .* gate.ε
        router_logits = router_logits .* jitter
    end
    
    return compute_gates(gate.base_gate, router_logits)
end