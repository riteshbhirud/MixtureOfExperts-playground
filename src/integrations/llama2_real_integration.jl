using Llama2
using LinearAlgebra
using ..MixtureOfExperts

import Llama2: ModelConfig, TransformerWeights, TransformerLayerWeights, 
               RunState, LanguageModel, load_karpathy_model, 
               rmsnorm!, matmul!, rope!, attention_weights!, combine_values!, 
               softmax!, sample

export MoETransformerWeights, MoELanguageModel, moe_transformer!, 
       create_moe_from_dense_weights, load_moe_model, sample_with_moe

"""
    MoETransformerWeights

Extended TransformerLayerWeights where dense FFN (w1, w2, w3) is replaced with MoE
"""
struct MoETransformerLayerWeights
    rms_att_weight::Vector{Float32}
    wq::AbstractMatrix
    wk::AbstractMatrix 
    wv::AbstractMatrix
    wo::AbstractMatrix
    
    # ... here FFN replaced with MoE
    rms_ffn_weight::Vector{Float32}
    moe_layer::MoELayer
    
    original_w1::Union{AbstractMatrix, Nothing}
    original_w2::Union{AbstractMatrix, Nothing}
    original_w3::Union{AbstractMatrix, Nothing}
end

struct MoETransformerWeights
    token_embedding_table::AbstractMatrix
    layers::Vector{MoETransformerLayerWeights}
    rms_final_weight::Vector{Float32}
    output_weight::AbstractMatrix
end

"""
    MoELanguageModel

Language model with MoE layers instead of dense FFN
"""
struct MoELanguageModel{TOK<:Llama2.Tokenizer}
    config::ModelConfig
    tokenizer::TOK
    weights::MoETransformerWeights
    training_stats::Dict{Symbol, Any}
end

function Base.show(io::IO, mime::MIME"text/plain", model::MoELanguageModel)
    println(io, "MoELanguageModel(")
    show(io, mime, model.config)
    num_experts = model.weights.layers[1].moe_layer.config.num_experts
    expert_type = model.weights.layers[1].moe_layer.config.expert_type
    println(io, ", experts=$(num_experts), type=$(expert_type)")
    print(io, ")")
end

"""
    moe_transformer!(token::Int, pos::Int, config::ModelConfig, s::RunState, weights::MoETransformerWeights; training::Bool = false)

Modified transformer forward pass with MoE layers replacing dense FFN.
Based on Llama2.jl src/inference.jl transformer! function.

**Key Change**: Replaces the FFN section:
```julia
# OLD (Dense FFN):
matmul!(s.hb, w.w1, s.xb)     # gate projection  
matmul!(s.hb2, w.w3, s.xb)    # up projection
s.hb .*= s.hb2                # element-wise gating
matmul!(s.xb, w.w2, s.hb)     # down projection

# NEW (MoE):
input_matrix = reshape(s.xb, :, 1)
moe_output, moe_loss = w.moe_layer(input_matrix; training=training)
s.xb = vec(moe_output)
```
"""
function moe_transformer!(token::Int, pos::Int, config::ModelConfig, s::RunState, 
                         weights::MoETransformerWeights; training::Bool = false)
    x = s.x
    
    (; dim, n_layers, n_heads, n_kv_heads) = config
    head_size = dim ÷ n_heads
    
    Llama2.dequantize!(x, weights.token_embedding_table[:, token])
    
    total_moe_loss = 0.0f0
    
    for l in 1:n_layers
        w = weights.layers[l]
        kv = s.kvcache_layers[l]
        
        rmsnorm!(s.xb, x, w.rms_att_weight)
        
        matmul!(s.q, w.wq, s.xb)
        matmul!(s.k, w.wk, s.xb) 
        matmul!(s.v, w.wv, s.xb)
        
        q = reshape(s.q, head_size, n_heads)
        k = reshape(s.k, head_size, n_kv_heads)
        
        rope!(q, pos, config)
        rope!(k, pos, config)
        
        copyto!(kv.key_cache[:, :, pos], s.k)
        copyto!(kv.value_cache[pos, :, :], s.v)
        
        att = reshape(s.att[1:(n_heads*pos)], pos, n_heads)
        attention_weights!(att, kv.key_cache, q)
        att ./= sqrt(Float32(head_size))
        
        for h in 1:n_heads
            softmax!(att[:, h])
        end
        
        xb = reshape(s.xb, head_size, n_heads)
        combine_values!(xb, kv.value_cache, att)
        
        matmul!(s.xb2, w.wo, s.xb)
        
        x .+= s.xb2
        
        rmsnorm!(s.xb, x, w.rms_ffn_weight)
        
        input_matrix = reshape(s.xb, :, 1)  
        
        moe_output, moe_loss = w.moe_layer(input_matrix; training=training)
        
        moe_vec = vec(moe_output)
        if length(moe_vec) == length(s.xb)
          s.xb .= moe_vec  # In-place assignment (more efficient)
        else
          resize!(s.xb, length(moe_vec))  # Handle size mismatch gracefully
          s.xb .= moe_vec
        end
        
        if training
            total_moe_loss += moe_loss
        end
        
        x .+= s.xb
    end
    
    rmsnorm!(x, x, weights.rms_final_weight)
    
    matmul!(s.logits, weights.output_weight, x)
    
    return total_moe_loss
end

"""
    create_moe_from_dense_weights(dense_weights::TransformerLayerWeights, moe_config::MoEConfig; initialize_from_dense::Bool = false)

Convert dense FFN weights to MoE layer weights.
Following mentor's instruction: "multiple smaller experts instead of one big FFN"
"""
function create_moe_from_dense_weights(dense_weights::TransformerLayerWeights, 
                                     moe_config::MoEConfig; 
                                     initialize_from_dense::Bool = false)
    
    moe_layer = MoELayer(moe_config)
    
    if initialize_from_dense && moe_config.expert_type == :gated
        initialize_experts_from_dense!(moe_layer, dense_weights, moe_config)
    end
    
    return MoETransformerLayerWeights(
        dense_weights.rms_att_weight,  
        dense_weights.wq,
        dense_weights.wk, 
        dense_weights.wv,
        dense_weights.wo,
        dense_weights.rms_ffn_weight, 
        moe_layer,                     
        dense_weights.w1,              
        dense_weights.w2,
        dense_weights.w3
    )
end

"""
    initialize_experts_from_dense!(moe_layer::MoELayer, dense_weights::TransformerLayerWeights, config::MoEConfig)

Initialize MoE experts using weights from the original dense FFN.
Implements "upcycling" technique from recent MoE research.
"""
function initialize_experts_from_dense!(moe_layer::MoELayer, dense_weights::TransformerLayerWeights, config::MoEConfig)
    if config.expert_type != :gated
        @warn "Dense initialization only supported for gated experts"
        return
    end
    
    num_experts = config.num_experts
    
    for (i, expert) in enumerate(moe_layer.experts)
        if expert isa GatedExpert
            scale_factor = 1.0f0 / sqrt(Float32(num_experts))
            
            expert.w1.weight .= dense_weights.w1 .* scale_factor
            expert.w2.weight .= dense_weights.w2 .* scale_factor  
            expert.w3.weight .= dense_weights.w3 .* scale_factor
            
            noise_scale = 0.01f0
            expert.w1.weight .+= randn(Float32, size(expert.w1.weight)) .* noise_scale
            expert.w2.weight .+= randn(Float32, size(expert.w2.weight)) .* noise_scale
            expert.w3.weight .+= randn(Float32, size(expert.w3.weight)) .* noise_scale
        end
    end
end

"""
    load_moe_model(checkpoint_filename::String, tokenizer_filename::String, moe_config::MoEConfig; initialize_from_dense::Bool = false)

Load Llama model and convert dense FFN layers to MoE.
This is the main entry point following mentor's instructions.
"""
function load_moe_model(checkpoint_filename::String, tokenizer_filename::String, 
                       moe_config::MoEConfig; initialize_from_dense::Bool = false)
    original_model = load_karpathy_model(checkpoint_filename, tokenizer_filename)
    
    moe_layers = MoETransformerLayerWeights[]
    for layer_weights in original_model.weights.layers
        moe_layer_weights = create_moe_from_dense_weights(layer_weights, moe_config; 
                                                         initialize_from_dense)
        push!(moe_layers, moe_layer_weights)
    end
    
    moe_weights = MoETransformerWeights(
        original_model.weights.token_embedding_table,
        moe_layers,
        original_model.weights.rms_final_weight,
        original_model.weights.output_weight
    )
    
    stats = Dict{Symbol, Any}(
        :total_moe_loss => 0.0f0,
        :expert_usage => zeros(Int, moe_config.num_experts, length(moe_layers)),
        :routing_entropy => Float32[]
    )
    
    return MoELanguageModel(original_model.config, original_model.tokenizer, moe_weights, stats)
end

"""
    sample_with_moe(model::MoELanguageModel, prompt::String = ""; kwargs...)

Sample from MoE model. Drop-in replacement for Llama2.sample() with MoE support.
"""
function sample_with_moe(model::MoELanguageModel, prompt::String = "";
                        temperature::Float32 = 0.9f0,
                        stop_on_special_token = true,
                        max_seq_len = typemax(Int),
                        bos_token = true,
                        training::Bool = false)
    
    if !bos_token && isempty(prompt)
        error("Prompt cannot be empty if bos_token = false")
    end
    
    (; config, weights, tokenizer) = model
    prompt_tokens = Llama2.encode(prompt, tokenizer)
    state = Llama2.RunState(config)
    
    time_start = time_ns()
    
    if bos_token
        pushfirst!(prompt_tokens, tokenizer.bos_token_id)
    end
    
    if !bos_token
        print(tokenizer.id_to_token[prompt_tokens[1]])
    end
    
    token = prompt_tokens[1]
    generated_seq_len = 0
    total_moe_loss = 0.0f0
    
    for pos in 1:min(config.seq_len, max_seq_len)
        moe_loss = moe_transformer!(token, pos, config, state, weights; training=training)
        total_moe_loss += moe_loss
        generated_seq_len += 1
        
        if pos + 1 <= length(prompt_tokens)
            next = prompt_tokens[pos + 1]
        else
            if temperature == 0f0
                next = argmax(state.logits)
            else
                state.logits ./= temperature
                softmax!(state.logits)
                next = Llama2.wsample(1:config.vocab_size, state.logits)
            end
        end
        
        if stop_on_special_token && (next == tokenizer.bos_token_id || next == tokenizer.eos_token_id)
            break
        end
        
        next_str = tokenizer.id_to_token[next]
        print(next_str)
        
        token = next
    end
    
    println()
    
    time_end = time_ns()
    tok_per_sec = generated_seq_len / (time_end - time_start) * 1e9
    
    if training
        avg_moe_loss = total_moe_loss / generated_seq_len
        @printf "achieved tok/s: %.2f, avg MoE loss: %.6f\n" tok_per_sec avg_moe_loss
    else
        @printf "achieved tok/s: %.2f\n" tok_per_sec
    end
    
    return nothing
end

"""
    create_simple_moe_config(config::ModelConfig; num_experts::Int = 8, use_random::Bool = true)

Create MoE config following mentor's progression:
1. Start with random gating ("at the beginning, just choose random expert")
2. Progress to sophisticated gating (Stanford CS336)
"""
function create_simple_moe_config(config::ModelConfig; 
                                 num_experts::Int = 8, 
                                 top_k::Int = 2,
                                 use_random::Bool = true,
                                 use_stanford_cs336::Bool = false)
    
    if use_random
        gate_type = RandomGating(top_k)
        balance_loss = NoBalancingLoss()
        @info "Using random gating as requested by mentor"
    elseif use_stanford_cs336  
        gate_type = TopKGating(top_k)
        balance_loss = SwitchTransformerLoss(0.01f0)  
        @info "Using Stanford CS336 Top-K gating with Switch Transformer loss"
    else
        gate_type = TopKGating(top_k) 
        balance_loss = NoBalancingLoss()
    end
    
    return MoEConfig(
        num_experts = num_experts,
        expert_type = :gated,                       
        input_dim = config.dim,
        hidden_dim = config.hidden_dim ÷ num_experts, 
        output_dim = config.dim,
        activation = x -> x * (1.0f0 / (1.0f0 + exp(-x))),
        gate_type = gate_type,
        top_k = top_k,
        balance_loss = balance_loss,
        use_fp32_router = true,                      
        noise_scale = use_stanford_cs336 ? 0.01f0 : 0.0f0,
        expert_dropout = 0.0f0
    )
end

"""
    analyze_moe_performance(model::MoELanguageModel, test_prompts::Vector{String})

Analyze MoE model performance and expert usage.
"""
function analyze_moe_performance(model::MoELanguageModel, test_prompts::Vector{String})
    println("=== MoE Performance Analysis ===")
    
    original_params = model.config.dim * model.config.hidden_dim * 3 * model.config.n_layers  
    moe_params = sum(sum(length, Flux.params(layer.moe_layer)) for layer in model.weights.layers)
    
    println("Original FFN parameters (estimated): $(original_params)")
    println("MoE parameters: $(moe_params)")
    println("Parameter reduction: $(round((1 - moe_params/original_params) * 100, digits=1))%")
    
    println("\n=== Generation Test ===")
    for (i, prompt) in enumerate(test_prompts)
        println("Prompt $i: \"$prompt\"")
        print("Output: ")
        sample_with_moe(model, prompt; max_seq_len=50, training=false)
        println()
    end
    
    if haskey(model.training_stats, :expert_usage)
        usage = model.training_stats[:expert_usage]
        println("\n=== Expert Usage Analysis ===")
        for layer in 1:size(usage, 2)
            layer_usage = usage[:, layer]
            balance_score = load_balance_score(layer_usage)
            println("Layer $layer balance score: $(round(balance_score, digits=3)) (1.0 = perfect)")
        end
    end
end

export create_simple_moe_config, analyze_moe_performance
