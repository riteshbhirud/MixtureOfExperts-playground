"""
    Router - Neural network for computing expert assignments
"""
struct Router{W, N, G}
    weight::W             
    noise_weight::N      
    gate_type::G        
    noise_scale::Float32
    use_fp32::Bool      
end

Flux.@functor Router (weight, noise_weight)

function Router(input_dim::Int, num_experts::Int, gate_type::GatingMechanism;
                noise_scale::Float32 = 0.0f0,
                use_noise_network::Bool = false,
                use_fp32::Bool = true)
    weight = Dense(input_dim, num_experts, bias=false)
    
    noise_weight = use_noise_network ? Dense(input_dim, num_experts, bias=false) : nothing
    
    return Router(weight, noise_weight, gate_type, noise_scale, use_fp32)
end

function (router::Router)(x::AbstractMatrix; training::Bool = false)
    if router.use_fp32 && eltype(x) != Float32
        x = Float32.(x)
    end
    
    router_logits = router.weight(x)
    
    if training && router.noise_scale > 0
        if !isnothing(router.noise_weight)
            noise_logits = router.noise_weight(x)
            noise = randn(Float32, size(router_logits)) .* softplus.(noise_logits)
            router_logits = router_logits .+ router.noise_scale .* noise
        else
            noise = randn(Float32, size(router_logits)) .* router.noise_scale
            router_logits = router_logits .+ noise
        end
    end
    
    expert_indices, expert_gates, router_probs = compute_gates(router.gate_type, router_logits)
    
    return expert_indices, expert_gates, router_probs, router_logits
end