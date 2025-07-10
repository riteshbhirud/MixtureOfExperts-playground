
using Transformers
using Transformers.HuggingFace
using Transformers.Layers
using ..MixtureOfExperts
using Flux

import Transformers.HuggingFace: HGFLlamaModel, HGFLlamaForCausalLM, HGFLlamaPreTrainedModel,
                                load_model, get_state_dict, HGFLlamaConfig

export MoEHGFLlamaModel, MoEHGFLlamaForCausalLM, LlamaMoEBlock, 
       load_moe_hgf_model, convert_hgf_to_moe

"""
    LlamaMoEBlock

Drop-in replacement for Llama FFN block using MoE.
Replaces: Layers.Chain(LLamaGated(gate, up), down) with MoE equivalent.
"""
struct LlamaMoEBlock{M, S} <: Layers.LayerStruct
    moe_layer::M
    training_mode::S
end

Flux.@functor LlamaMoEBlock (moe_layer,)

function LlamaMoEBlock(moe_config::MoEConfig)
    training_mode = Ref(false)
    moe_layer = MoELayer(moe_config)
    return LlamaMoEBlock(moe_layer, training_mode)
end

Layers.argument_names(::LlamaMoEBlock) = (:hidden_state,)

function (block::LlamaMoEBlock)(nt::NamedTuple)
    x = nt.hidden_state
    
    original_shape = size(x)
    
    if ndims(x) == 3
        hidden_dim, seq_len, batch_size = original_shape
        x_reshaped = reshape(x, hidden_dim, seq_len * batch_size)
    else
        x_reshaped = x
    end
    
    output, moe_loss = block.moe_layer(x_reshaped; training=block.training_mode[])
    
    if ndims(x) == 3
        output = reshape(output, original_shape)
    end
    
    result = (hidden_state = output, moe_loss = moe_loss)
    
    extra_fields = Base.structdiff(nt, NamedTuple{(:hidden_state,)})
    return merge(result, extra_fields)
end

"""
    MoEPreNormResidual

PreNormResidual wrapper that handles MoE loss accumulation.
Replaces Layers.PreNormResidual for FFN blocks.
"""
struct MoEPreNormResidual{L, N} <: Layers.LayerStruct
    layer::L  
    norm::N   
end

Flux.@functor MoEPreNormResidual (layer, norm)

Layers.argument_names(prenr::MoEPreNormResidual) = Layers.argument_names(prenr.layer)

function (prenr::MoEPreNormResidual)(nt::NamedTuple)
    norm_nt = Layers.apply_on_namedtuple(prenr.norm, nt)
    
    moe_nt = Layers.apply_on_namedtuple(prenr.layer, norm_nt)
    
    hidden_state = moe_nt.hidden_state + nt.hidden_state
    
    result = Layers.return_hidden_state(nt, hidden_state)
    
    if haskey(moe_nt, :moe_loss)
        result = merge(result, (moe_loss = moe_nt.moe_loss,))
    end
    
    return result
end

"""
    MoETransformerBlock

Transformer block with MoE FFN. Replaces standard TransformerBlock.
"""
struct MoETransformerBlock{A, F} <: Layers.AbstractTransformerBlock
    attention::A 
    feedforward::F 
end

Flux.@functor MoETransformerBlock

Layers.argument_names(b::MoETransformerBlock) = Base.merge_names(
    Layers.argument_names(b.attention), 
    Layers.argument_names(b.feedforward)
)

function (block::MoETransformerBlock)(nt::NamedTuple)
    attention_output = Layers.apply_on_namedtuple(block.attention, nt)
    final_output = Layers.apply_on_namedtuple(block.feedforward, attention_output)
    return final_output
end

"""
    MoEHGFLlamaModel

HuggingFace Llama model with MoE layers instead of dense FFN.
"""
struct MoEHGFLlamaModel{E, D}
    embed::E
    decoder::D
    moe_configs::Vector{MoEConfig}  
end

Flux.@functor MoEHGFLlamaModel

function (model::MoEHGFLlamaModel)(nt::NamedTuple)
    set_moe_training_mode!(model, get(nt, :training, false))
    
    embed_output = Layers.apply_on_namedtuple(model.embed, nt)
    decoder_output = Layers.apply_on_namedtuple(model.decoder, embed_output)
    
    total_moe_loss = accumulate_moe_losses(decoder_output)
    
    result = merge(decoder_output, (total_moe_loss = total_moe_loss,))
    return result
end

"""
    MoEHGFLlamaForCausalLM

HuggingFace Llama for Causal LM with MoE.
"""
struct MoEHGFLlamaForCausalLM{M, C}
    model::M  
    cls::C    
end

Flux.@functor MoEHGFLlamaForCausalLM

function (model::MoEHGFLlamaForCausalLM)(nt::NamedTuple)
    model_output = model.model(nt)
    cls_output = Layers.apply_on_namedtuple(model.cls, model_output)
    
    if haskey(model_output, :total_moe_loss)
        cls_output = merge(cls_output, (total_moe_loss = model_output.total_moe_loss,))
    end
    
    return cls_output
end

"""
    load_moe_hgf_model(model_name::String; num_experts::Int = 8, use_random::Bool = true, kwargs...)

Load HuggingFace Llama model and convert dense FFN to MoE.
Main entry point following mentor's instructions.
"""
function load_moe_hgf_model(model_name::String; 
                           num_experts::Int = 8, 
                           top_k::Int = 2,
                           use_random::Bool = true,
                           use_stanford_cs336::Bool = false,
                           expert_type::Symbol = :gated,
                           kwargs...)
    
    original_model = HuggingFace.load_model(model_name; kwargs...)
    config = HuggingFace.load_config(model_name; kwargs...)
    
    moe_config = create_hgf_moe_config(config; 
                                      num_experts, 
                                      top_k, 
                                      use_random, 
                                      use_stanford_cs336,
                                      expert_type)
    
    if original_model isa HGFLlamaForCausalLM
        moe_model = convert_hgf_causal_lm_to_moe(original_model, moe_config)
    elseif original_model isa HGFLlamaModel  
        moe_model = convert_hgf_model_to_moe(original_model, moe_config)
    else
        error("Unsupported model type: $(typeof(original_model))")
    end
    
    @info "Converted $(typeof(original_model)) to MoE with $(num_experts) experts"
    return moe_model
end

"""
    convert_hgf_model_to_moe(model::HGFLlamaModel, moe_config::MoEConfig)

Convert HGFLlamaModel dense FFN layers to MoE.
"""
function convert_hgf_model_to_moe(model::HGFLlamaModel, moe_config::MoEConfig)
    embed = model.embed
    
    if model.decoder isa Layers.Chain{<:Tuple{Layers.Transformer, Any}}
        transformer, final_ln = model.decoder.layers
        moe_transformer = convert_transformer_to_moe(transformer, moe_config)
        decoder = Layers.Chain(moe_transformer, final_ln)
    else
        error("Unexpected decoder structure: $(typeof(model.decoder))")
    end
    
    n_layers = length(moe_transformer.blocks)
    moe_configs = [moe_config for _ in 1:n_layers]
    
    return MoEHGFLlamaModel(embed, decoder, moe_configs)
end

"""
    convert_hgf_causal_lm_to_moe(model::HGFLlamaForCausalLM, moe_config::MoEConfig)

Convert HGFLlamaForCausalLM to MoE version.
"""
function convert_hgf_causal_lm_to_moe(model::HGFLlamaForCausalLM, moe_config::MoEConfig)
    moe_base_model = convert_hgf_model_to_moe(model.model, moe_config)
    return MoEHGFLlamaForCausalLM(moe_base_model, model.cls)
end

"""
    convert_transformer_to_moe(transformer::Layers.Transformer, moe_config::MoEConfig)

Convert Transformer blocks to use MoE FFN layers.
"""
function convert_transformer_to_moe(transformer::Layers.Transformer, moe_config::MoEConfig)
    moe_blocks = []
    
    for block in transformer.blocks
        if block isa Layers.TransformerBlock
            attention = block.attention 
            
            if block.feedforward isa Layers.PreNormResidual
                norm = block.feedforward.norm
                moe_block = LlamaMoEBlock(moe_config)
                feedforward = MoEPreNormResidual(moe_block, norm)
            else
                error("Unexpected feedforward structure: $(typeof(block.feedforward))")
            end
            
            moe_transformer_block = MoETransformerBlock(attention, feedforward)
            push!(moe_blocks, moe_transformer_block)
        else
            error("Unexpected block type: $(typeof(block))")
        end
    end
    
    return Layers.Transformer(Tuple(moe_blocks), transformer.f)
end

"""
    create_hgf_moe_config(hf_config; kwargs...)

Create MoE config from HuggingFace config following mentor's progression.
"""
function create_hgf_moe_config(hf_config; 
                              num_experts::Int = 8,
                              top_k::Int = 2, 
                              use_random::Bool = true,
                              use_stanford_cs336::Bool = false,
                              expert_type::Symbol = :gated)
    
    input_dim = hf_config[:hidden_size]
    hidden_dim = hf_config[:intermediate_size] ÷ num_experts 
    activation = get(HuggingFace.ACT2FN, Symbol(hf_config[:hidden_act]), silu)
    
    if use_random
        gate_type = RandomGating(top_k)
        balance_loss = NoBalancingLoss()
        @info "Creating random gating MoE as requested by mentor"
    elseif use_stanford_cs336
        gate_type = TopKGating(top_k)
        balance_loss = SwitchTransformerLoss(0.01f0) 
        @info "Creating Stanford CS336 Top-K gating with Switch Transformer loss"
    else
        gate_type = TopKGating(top_k)
        balance_loss = NoBalancingLoss()
    end
    
    return MoEConfig(
        num_experts = num_experts,
        expert_type = expert_type,
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        output_dim = input_dim,
        activation = activation,
        gate_type = gate_type,
        top_k = top_k,
        balance_loss = balance_loss,
        use_fp32_router = true,  
        noise_scale = use_stanford_cs336 ? 0.01f0 : 0.0f0,
        expert_dropout = 0.0f0
    )
end

"""
    Helper functions for MoE loss handling
"""
function set_moe_training_mode!(model::MoEHGFLlamaModel, training::Bool)
    for block in model.decoder[1].blocks
        if block isa MoETransformerBlock && block.feedforward.layer isa LlamaMoEBlock
            block.feedforward.layer.training_mode[] = training
        end
    end
end

function accumulate_moe_losses(output::NamedTuple)
    total_loss = 0.0f0
    
    for (key, value) in pairs(output)
        if key == :moe_loss && value isa Number
            total_loss += value
        elseif value isa NamedTuple
            total_loss += accumulate_moe_losses(value)
        elseif value isa Tuple
            for item in value
                if item isa NamedTuple && haskey(item, :moe_loss)
                    total_loss += item.moe_loss
                end
            end
        end
    end
    
    return total_loss
end

"""
    analyze_hgf_moe_model(model::Union{MoEHGFLlamaModel, MoEHGFLlamaForCausalLM})

Analyze HuggingFace MoE model performance and statistics.
"""
function analyze_hgf_moe_model(model::Union{MoEHGFLlamaModel, MoEHGFLlamaForCausalLM})
    println("=== HuggingFace MoE Model Analysis ===")
    
    base_model = model isa MoEHGFLlamaForCausalLM ? model.model : model
    
    moe_params = sum(Flux.params(base_model)) do p
        length(p)
    end
    
    first_moe_config = base_model.moe_configs[1]
    dense_hidden_dim = first_moe_config.hidden_dim * first_moe_config.num_experts
    estimated_dense_params = first_moe_config.input_dim * dense_hidden_dim * 3 * length(base_model.moe_configs)
    
    println("Estimated original dense parameters: $(estimated_dense_params)")
    println("MoE parameters: $(moe_params)")
    println("Parameter reduction: $(round((1 - moe_params/estimated_dense_params) * 100, digits=1))%")
    
    config = first_moe_config
    println("Expert configuration:")
    println("  - Number of experts: $(config.num_experts)")
    println("  - Expert type: $(config.expert_type)")
    println("  - Top-K: $(config.top_k)")
    println("  - Gating: $(typeof(config.gate_type))")
    println("  - Balance loss: $(typeof(config.balance_loss))")
    
    return (
        moe_params = moe_params,
        estimated_dense_params = estimated_dense_params,
        parameter_reduction = (1 - moe_params/estimated_dense_params) * 100,
        config = config
    )
end

if !@isdefined(silu)
    silu(x) = x * sigmoid(x)
end

export create_hgf_moe_config, analyze_hgf_moe_model, set_moe_training_mode!