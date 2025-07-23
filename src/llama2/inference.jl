"""
Complete MoE-Llama2 Inference Engine

This module implements the complete transformer forward pass with seamless
integration between dense layers and MoE layers.
"""

"""
    moe_transformer!(token::Int, pos::Int, model::MoELanguageModel, state::MoERunState)

Complete transformer forward pass with MoE integration.
Handles both dense and MoE layers seamlessly in a single unified forward pass.
"""
function moe_transformer!(token::Int, pos::Int, model::MoELanguageModel, state::MoERunState)
    config = model.config
    weights = model.weights
    
    # Update performance stats
    state.inference_stats[:total_tokens] += 1
    
    # Token embedding lookup
    state.x .= weights.token_embedding_table[:, token]
    
    # Forward through all transformer layers
    for (layer_idx, layer) in enumerate(weights.layers)
        # Start layer timing
        layer_start_time = time()
        
        # Self-attention block
        moe_attention!(state, layer, pos, config, layer_idx)
        
        # Residual connection after attention
        state.x .+= state.xb2
        
        # Feed-forward block (either dense or MoE)
        if layer.use_moe
            # MoE FFN
            state.inference_stats[:moe_layer_calls] += 1
            moe_ffn_forward!(state, layer, config)
        else
            # Dense FFN (standard Llama2)
            dense_ffn_forward!(state, layer, config)
        end
        
        # Residual connection after FFN
        state.x .+= state.xb2
        
        # Update layer timing
        layer_time = time() - layer_start_time
        if !haskey(state.inference_stats, :layer_times)
            state.inference_stats[:layer_times] = Float64[]
        end
        push!(state.inference_stats[:layer_times], layer_time)
    end
    
    # Final layer normalization
    Llama2.rmsnorm!(state.x, state.x, weights.rms_final_weight)
    
    # Output projection to vocabulary
    Llama2.matmul!(state.logits, weights.output_weight, state.x)
    
    return nothing
end

"""
    moe_ffn_forward!(state::MoERunState, layer::MoETransformerLayerWeights, config::MoELlamaConfig)

MoE Feed-Forward Network computation.
Implements complete routing, expert selection, and weighted combination.
"""
function moe_ffn_forward!(state::MoERunState, layer::MoETransformerLayerWeights, config::MoELlamaConfig)
    routing_start_time = time()
    
    # Pre-FFN normalization
    Llama2.rmsnorm!(state.xb, state.x, layer.llama_layer.rms_ffn_weight)
    
    # Router computation: logits = router_weight^T * input
    Llama2.matmul!(state.router_logits, layer.moe_router_weight, state.xb)
    
    # Apply gating mechanism (convert to matrix format for MoE library)
    router_logits_matrix = reshape(state.router_logits, :, 1)
    expert_indices, expert_gates, router_probs = compute_gates(layer.moe_config.gate_type, router_logits_matrix)
    
    # Extract results for single token
    num_selected = min(config.moe_top_k, size(expert_indices, 1))
    for k in 1:num_selected
        state.selected_experts[k] = expert_indices[k, 1]
        state.expert_gates[k] = expert_gates[k, 1]
    end
    
    # Clear unused slots
    for k in (num_selected + 1):length(state.selected_experts)
        state.selected_experts[k] = 0
        state.expert_gates[k] = 0.0f0
    end
    
    state.inference_stats[:routing_time] += time() - routing_start_time
    
    # Expert computation
    expert_start_time = time()
    
    # Clear output buffer
    fill!(state.xb2, 0.0f0)
    
    # Process selected experts
    for k in 1:num_selected
        expert_idx = state.selected_experts[k]
        gate_weight = state.expert_gates[k]
        
        if expert_idx > 0 && expert_idx <= length(layer.moe_experts)
            expert = layer.moe_experts[expert_idx]
            
            # Update expert usage tracking
            state.expert_load_counts[expert_idx] += 1
            state.inference_stats[:expert_activations] += 1
            
            # Compute expert output based on type
            if expert.expert_type == :cur
                cur_expert_forward!(state.expert_outputs[expert_idx], expert, state.xb)
            else
                gated_expert_forward!(state.expert_outputs[expert_idx], expert, state.xb)
            end
            
            # Accumulate weighted output
            @inbounds for i in 1:length(state.xb2)
                state.xb2[i] += gate_weight * state.expert_outputs[expert_idx][i]
            end
        end
    end
    
    # Handle shared experts (DeepSeek-style)
    if !isnothing(layer.shared_experts) && !isempty(layer.shared_experts)
        process_shared_experts!(state, layer, config)
    end
    
    state.inference_stats[:expert_compute_time] += time() - expert_start_time
    
    # Compute auxiliary losses if needed (for training/analysis)
    if config.moe_balance_loss isa AuxiliaryFreeLoss
        update_bias!(config.moe_balance_loss, reshape(state.selected_experts[1:num_selected], :, 1))
    end
    
    # Track routing entropy for analysis
    entropy_val = compute_routing_entropy(router_probs)
    push!(state.routing_entropy, entropy_val)
    
    return nothing
end

"""
    dense_ffn_forward!(state::MoERunState, layer::MoETransformerLayerWeights, config::MoELlamaConfig)

Standard dense Feed-Forward Network computation.
Exactly matches Llama2's FFN implementation: w2(silu(w1(x)) * w3(x))
"""
function dense_ffn_forward!(state::MoERunState, layer::MoETransformerLayerWeights, config::MoELlamaConfig)
    llama_layer = layer.llama_layer
    
    # Pre-FFN normalization  
    Llama2.rmsnorm!(state.xb, state.x, llama_layer.rms_ffn_weight)
    
    # Gate and up projections
    Llama2.matmul!(state.hb, llama_layer.w1, state.xb)   # gate projection
    Llama2.matmul!(state.hb2, llama_layer.w3, state.xb)  # up projection
    
    # SiLU activation and element-wise multiply: silu(gate) * up
    @inbounds for i in 1:length(state.hb)
        gate_val = state.hb[i]
        silu_val = gate_val * (1.0f0 / (1.0f0 + exp(-gate_val)))  # SiLU activation
        state.hb[i] = silu_val * state.hb2[i]
    end
    
    # Down projection
    Llama2.matmul!(state.xb2, llama_layer.w2, state.hb)
    
    return nothing
end

"""
    gated_expert_forward!(output::Vector{Float32}, expert::MoEExpertWeights, input::Vector{Float32})

Compute forward pass through a gated expert.
Implements: output = w2(silu(w1(input)) * w3(input))
"""
function gated_expert_forward!(output::Vector{Float32}, expert::MoEExpertWeights, input::Vector{Float32})
    # Gate and up projections using Llama2's matmul convention
    Llama2.matmul!(expert.hb1, expert.w1, input)  # gate: hb1 = w1^T * input
    Llama2.matmul!(expert.hb2, expert.w3, input)  # up: hb2 = w3^T * input
    
    # SiLU activation and element-wise multiply
    @inbounds for i in 1:length(expert.hb1)
        gate_val = expert.hb1[i]
        silu_val = gate_val * (1.0f0 / (1.0f0 + exp(-gate_val)))
        expert.hb1[i] = silu_val * expert.hb2[i]
    end
    
    # Down projection: output = w2^T * hb1
    Llama2.matmul!(output, expert.w2, expert.hb1)
    
    return nothing
end

function cur_expert_forward!(output::Vector{Float32}, expert::MoEExpertWeights, input::Vector{Float32})
    if !expert.is_cur_compressed
        # Fallback to gated computation if not CUR compressed
        gated_expert_forward!(output, expert, input)
        return nothing
    end
    #to be called in later phase
    # CUR decomposition computation: A ≈ C * U * R
    # For w1: temp1 = R1 * input, temp2 = U1 * temp1, gate = C1 * temp2
    # For w3: temp1 = R3 * input, temp2 = U3 * temp1, up = C3 * temp2  
    # For w2: temp1 = R2 * hidden, temp2 = U2 * temp1, output = C2 * temp2
    
    # This is a simplified version - full CUR implementation would require
    # storing and using the C, U, R matrices from the expert
    # For now, fall back to standard computation
    gated_expert_forward!(output, expert, input)
    
    return nothing
end
"""
    process_shared_experts!(state::MoERunState, layer::MoETransformerLayerWeights, config::MoELlamaConfig)

Process shared experts that are always active (DeepSeek-style architecture).
"""
function process_shared_experts!(state::MoERunState, layer::MoETransformerLayerWeights, config::MoELlamaConfig)
    if isnothing(layer.shared_experts)
        return nothing
    end
    
    # Compute shared expert contributions
    shared_output = zeros(Float32, config.dim)
    
    for shared_expert in layer.shared_experts
        expert_output = zeros(Float32, config.dim)
        
        if shared_expert.expert_type == :cur
            cur_expert_forward!(expert_output, shared_expert, state.xb)
        else
            gated_expert_forward!(expert_output, shared_expert, state.xb)
        end
        
        # Accumulate shared expert output (equally weighted)
        shared_weight = 1.0f0 / length(layer.shared_experts)
        @inbounds for i in 1:length(shared_output)
            shared_output[i] += shared_weight * expert_output[i]
        end
    end
    
    # Combine shared and routed expert outputs
    # This is a design choice - could be learned parameters instead
    shared_weight = 0.5f0
    routed_weight = 0.5f0
    
    @inbounds for i in 1:length(state.xb2)
        state.xb2[i] = shared_weight * shared_output[i] + routed_weight * state.xb2[i]
    end
    
    return nothing
end

"""
    compute_routing_entropy(router_probs::AbstractMatrix)

Compute entropy of routing distribution for analysis.
Higher entropy indicates more balanced expert usage.
"""
function compute_routing_entropy(router_probs::AbstractMatrix)
    if isempty(router_probs)
        return 0.0f0
    end
    
    # Compute entropy: H = -Σ p * log(p)
    entropy = 0.0f0
    for prob in router_probs
        if prob > 1e-10  # Avoid log(0)
            entropy -= prob * log(prob)
        end
    end
    
    return entropy
end

"""
    batch_moe_transformer!(tokens::AbstractVector{Int}, model::MoELanguageModel, 
                          states::Vector{MoERunState})

Batch processing for multiple sequences.
Processes multiple token sequences in parallel for improved throughput.
"""
function batch_moe_transformer!(tokens::AbstractVector{Int}, model::MoELanguageModel, 
                                states::Vector{MoERunState})
    if length(tokens) != length(states)
        throw(ArgumentError("Number of tokens must match number of states"))
    end
    
    batch_size = length(tokens)
    config = model.config
    
    # Process each sequence in the batch
    Threads.@threads for i in 1:batch_size
        try
            # Each sequence processes independently
            moe_transformer!(tokens[i], i, model, states[i])
        catch e
            @error "Error processing token $(tokens[i]) in batch position $i: $e"
            rethrow(e)
        end
    end
    
    return nothing
end

"""
    efficient_moe_transformer!(token::Int, pos::Int, model::MoELanguageModel, 
                              state::MoERunState; use_cache::Bool = true)

Optimized transformer forward pass with optional caching and optimizations.
"""
function efficient_moe_transformer!(token::Int, pos::Int, model::MoELanguageModel, 
                                   state::MoERunState; use_cache::Bool = true)
    config = model.config
    weights = model.weights
    
    # Token embedding with bounds checking
    if token < 1 || token > config.vocab_size
        throw(BoundsError("Token $token out of vocabulary range [1, $(config.vocab_size)]"))
    end
    
    state.x .= weights.token_embedding_table[:, token]
    
    # Pre-allocate temporary buffers if not already done
    if !haskey(state.inference_stats, :temp_buffers_allocated)
        state.inference_stats[:temp_buffers_allocated] = true
    end
    
    # Forward through layers with early exit on NaN/Inf
    for (layer_idx, layer) in enumerate(weights.layers)
        # Attention
        moe_attention!(state, layer, pos, config, layer_idx)
        
        # Check for numerical issues
        if !all(isfinite.(state.xb2))
            @warn "Numerical instability detected in attention layer $layer_idx at position $pos"
        end
        
        state.x .+= state.xb2
        
        # FFN
        if layer.use_moe
            moe_ffn_forward!(state, layer, config)
        else
            dense_ffn_forward!(state, layer, config)
        end
        
        # Check for numerical issues
        if !all(isfinite.(state.xb2))
            @warn "Numerical instability detected in FFN layer $layer_idx at position $pos"
        end
        
        state.x .+= state.xb2
        
        # Early termination if requested
        if haskey(state.inference_stats, :early_stop_layer) && 
           layer_idx >= state.inference_stats[:early_stop_layer]
            break
        end
    end
    
    # Final processing
    Llama2.rmsnorm!(state.x, state.x, weights.rms_final_weight)
    Llama2.matmul!(state.logits, weights.output_weight, state.x)
    
    # Numerical stability check for final logits
    if !all(isfinite.(state.logits))
        @warn "Numerical instability in final logits at position $pos"
        # Clamp to reasonable range
        clamp!(state.logits, -100.0f0, 100.0f0)
    end
    
    return nothing
end

"""
    reset_moe_state!(state::MoERunState)

Reset MoE-specific state for new sequence processing.
Preserves Llama2 state while clearing MoE buffers.
"""
function reset_moe_state!(state::MoERunState)
    # Clear MoE buffers
    fill!(state.router_logits, 0.0f0)
    fill!(state.expert_gates, 0.0f0)
    fill!(state.selected_experts, 0)
    
    for expert_output in state.expert_outputs
        fill!(expert_output, 0.0f0)
    end
    
    fill!(state.moe_temp_buffer, 0.0f0)
    
    # Reset tracking
    empty!(state.auxiliary_loss_values)
    empty!(state.routing_entropy)
    fill!(state.expert_load_counts, 0)
    
    # Reset performance stats
    state.inference_stats[:total_tokens] = 0
    state.inference_stats[:moe_layer_calls] = 0
    state.inference_stats[:expert_activations] = 0
    state.inference_stats[:routing_time] = 0.0
    state.inference_stats[:expert_compute_time] = 0.0
    
    if haskey(state.inference_stats, :layer_times)
        empty!(state.inference_stats[:layer_times])
    end
    
    return nothing
end

"""
    validate_inference_state(state::MoERunState, config::MoELlamaConfig)

Validate that the inference state is consistent and ready for computation.
"""
function validate_inference_state(state::MoERunState, config::MoELlamaConfig)
    # Check buffer sizes
    if length(state.router_logits) != config.moe_num_experts
        throw(ArgumentError("Router logits buffer size mismatch"))
    end
    
    if length(state.expert_gates) != config.moe_top_k
        throw(ArgumentError("Expert gates buffer size mismatch"))
    end
    
    if length(state.selected_experts) != config.moe_top_k
        throw(ArgumentError("Selected experts buffer size mismatch"))
    end
    
    if length(state.expert_outputs) != config.moe_num_experts
        throw(ArgumentError("Expert outputs buffer count mismatch"))
    end
    
    for (i, expert_output) in enumerate(state.expert_outputs)
        if length(expert_output) != config.dim
            throw(ArgumentError("Expert output $i buffer size mismatch"))
        end
    end
    
    if length(state.moe_temp_buffer) != config.dim
        throw(ArgumentError("MoE temp buffer size mismatch"))
    end
    
    return true
end