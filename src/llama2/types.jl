"""
Llama2 Integration Types

This file defines wrapper types that extend Llama2 functionality with MoE capabilities
without modifying the original Llama2 library.
"""

"""
    MoELlamaConfig

Extended configuration that wraps Llama2.ModelConfig with MoE settings.
Preserves all original Llama2 config fields while adding MoE-specific parameters.
"""
struct MoELlamaConfig
    # Core Llama2 config (wrapped, not modified)
    llama_config::Llama2.ModelConfig
    
    # MoE-specific configuration
    moe_layers::Vector{Int}                    # Which layers use MoE (e.g., [2, 4, 6])
    moe_num_experts::Int                       # Number of experts per MoE layer
    moe_top_k::Int                            # Top-k routing
    moe_expert_type::Symbol                   # :standard, :gated, :cur
    moe_gate_type::GatingMechanism            # Gating mechanism from MoE library
    moe_balance_loss::LoadBalancingLoss       # Load balancing strategy
    
    # Expert initialization settings
    expert_init_strategy::Symbol              # :copy, :perturb, :split, :random
    expert_init_noise::Float32               # Noise level for :perturb strategy
    
    # Advanced MoE settings
    use_shared_experts::Bool                  # DeepSeek-style shared experts
    num_shared_experts::Int                   # Number of always-active experts
    expert_dropout::Float32                   # Dropout within experts
    capacity_factor::Float32                  # For expert choice routing
    drop_tokens::Bool                        # Drop tokens when capacity exceeded
    
    # CUR decomposition settings (when moe_expert_type = :cur)
    use_cur::Bool                            # Enable CUR decomposition
    cur_rank::Union{Int, Nothing}            # CUR rank (nothing = auto)
    cur_oversample::Int                      # Oversampling for CUR
    
    # Performance settings
    use_fp32_router::Bool                    # Use FP32 for router (stability)
    router_jitter::Float32                   # Training jitter for router
    z_loss_weight::Float32                   # Z-loss weight for logit growth
end

# Property delegation: access llama_config fields directly
function Base.getproperty(config::MoELlamaConfig, name::Symbol)
    if hasfield(MoELlamaConfig, name)
        return getfield(config, name)
    else
        return getproperty(config.llama_config, name)
    end
end

"""
    MoEExpertWeights

Represents weights for a single MoE expert, matching Llama2's gated FFN structure.
Supports multiple expert types: standard, gated, and CUR-compressed.
"""
struct MoEExpertWeights{W1, W2, W3, B1, B2}
    # Core weight matrices (dimensions match Llama2 exactly)
    w1::W1              # Gate projection: (dim, hidden_dim)
    w2::W2              # Down projection: (hidden_dim, dim)  
    w3::W3              # Up projection: (dim, hidden_dim)
    
    # Temporary computation buffers (not saved/loaded)
    hb1::B1             # Hidden buffer 1: (hidden_dim,)
    hb2::B2             # Hidden buffer 2: (hidden_dim,)
    
    # Expert metadata
    expert_type::Symbol # :standard, :gated, :cur
    is_cur_compressed::Bool
    
    # CUR-specific fields (only used when expert_type = :cur)
    cur_c::Union{Nothing, AbstractMatrix}    # C matrix for CUR
    cur_u::Union{Nothing, AbstractMatrix}    # U matrix for CUR  
    cur_r::Union{Nothing, AbstractMatrix}    # R matrix for CUR
end

"""
    MoETransformerLayerWeights

Extended transformer layer that can be either dense (original Llama2) or MoE.
Uses composition to wrap original Llama2 layer while adding MoE capabilities.
"""
struct MoETransformerLayerWeights
    # Original Llama2 layer (preserved exactly)
    llama_layer::Llama2.TransformerLayerWeights
    
    # MoE configuration
    use_moe::Bool                           # Whether this layer uses MoE
    
    # MoE-specific weights (only used when use_moe = true)
    moe_experts::Union{Nothing, Vector{MoEExpertWeights}}      # Expert weights
    moe_router_weight::Union{Nothing, AbstractMatrix}         # Router: (dim, num_experts) 
    moe_config::Union{Nothing, MoEConfig}                     # MoE configuration
    
    # Shared expert weights (for DeepSeek-style architectures)
    shared_experts::Union{Nothing, Vector{MoEExpertWeights}}  # Always-active experts
    
    # Advanced MoE features
    auxiliary_loss_state::Union{Nothing, Dict{Symbol, Any}}   # For auxiliary-free balancing
    expert_usage_stats::Union{Nothing, Vector{Int}}          # Usage tracking
end

"""
    MoETransformerWeights  

Complete transformer weights with mixed dense/MoE layers.
Wraps Llama2.TransformerWeights while replacing specified layers with MoE.
"""
struct MoETransformerWeights
    # Global weights (unchanged from Llama2)
    token_embedding_table::AbstractMatrix   # (dim, vocab_size)
    rms_final_weight::Vector{Float32}       # (dim,)
    output_weight::AbstractMatrix           # (dim, vocab_size) - CRITICAL: Llama2 convention
    
    # Layer weights (mix of dense and MoE)
    layers::Vector{MoETransformerLayerWeights}
    
    # Model metadata
    config::MoELlamaConfig
    conversion_info::Dict{Symbol, Any}      # Track conversion from original model
end

"""
    MoEKVCache

Extended KV cache that preserves Llama2's caching strategy.
Identical to Llama2.KVCache but wrapped for type consistency.
"""
struct MoEKVCache
    # Llama2 KV cache (unchanged)
    llama_kv_cache::Llama2.KVCache
end

# Delegate all operations to underlying Llama2 cache
Base.getproperty(cache::MoEKVCache, name::Symbol) = getproperty(cache.llama_kv_cache, name)
Base.setproperty!(cache::MoEKVCache, name::Symbol, value) = setproperty!(cache.llama_kv_cache, name, value)

"""
    MoERunState

Extended run state that includes all Llama2 buffers plus MoE-specific buffers.
Preserves all Llama2 computation patterns while adding MoE routing state.
"""
struct MoERunState
    # Original Llama2 state (preserved exactly)
    llama_state::Llama2.RunState
    
    # MoE-specific computation buffers
    router_logits::Vector{Float32}           # Router output: (num_experts,)
    expert_gates::Vector{Float32}            # Selected gate weights: (top_k,)
    selected_experts::Vector{Int}            # Selected expert indices: (top_k,)
    expert_outputs::Vector{Vector{Float32}}  # Expert outputs: num_experts × (dim,)
    moe_temp_buffer::Vector{Float32}         # Temporary computation: (dim,)
    
    # Advanced MoE state
    auxiliary_loss_values::Vector{Float32}   # Load balancing losses
    routing_entropy::Vector{Float32}        # Routing entropy tracking
    expert_load_counts::Vector{Int}         # Expert usage counters
    
    # Performance tracking
    inference_stats::Dict{Symbol, Any}      # Performance metrics
end

# Delegate access to Llama2 state fields
function Base.getproperty(state::MoERunState, name::Symbol)
    if hasfield(MoERunState, name)
        return getfield(state, name)
    else
        return getproperty(state.llama_state, name)
    end
end

"""
    MoELanguageModel

Complete language model with MoE integration.
Wraps Llama2.LanguageModel while replacing FFN layers with MoE where specified.
"""
struct MoELanguageModel{TOK<:Llama2.Tokenizer}
    # Model components
    config::MoELlamaConfig
    tokenizer::TOK                          # Llama2 tokenizer (unchanged)
    weights::MoETransformerWeights
    
    # Model metadata
    original_model_info::Dict{String, Any}  # Info about source Llama2 model
    moe_conversion_info::Dict{String, Any}  # Details about MoE conversion
    
    # Performance optimization
    kv_cache_pool::Vector{MoEKVCache}       # Pre-allocated KV caches
    state_pool::Vector{MoERunState}         # Pre-allocated run states
end

# Delegate tokenizer access
Base.getproperty(model::MoELanguageModel, name::Symbol) = 
    hasfield(MoELanguageModel, name) ? getfield(model, name) : getproperty(model.tokenizer, name)

"""
    Expert creation utilities
"""

"""
    create_moe_expert_weights(config::MoELlamaConfig, expert_type::Symbol = :gated)

Create expert weights matching the specified type and configuration.
"""
function create_moe_expert_weights(config::MoELlamaConfig, expert_type::Symbol = :gated)
    dim = config.dim
    hidden_dim = config.hidden_dim
    
    # Initialize using He initialization (matches Llama2)
    σ = sqrt(2.0f0 / dim)
    
    if expert_type == :cur && config.use_cur
        # CUR-compressed expert
        rank = something(config.cur_rank, hidden_dim ÷ 4)
        
        # Create full matrices first
        w1_full = randn(Float32, dim, hidden_dim) .* σ
        w2_full = randn(Float32, hidden_dim, dim) .* σ  
        w3_full = randn(Float32, dim, hidden_dim) .* σ
        
        # Apply CUR decomposition
        w1_cur = cur_decompose(w1_full; rank=rank, oversample=config.cur_oversample)
        w2_cur = cur_decompose(w2_full; rank=rank, oversample=config.cur_oversample)
        w3_cur = cur_decompose(w3_full; rank=rank, oversample=config.cur_oversample)
        
        return MoEExpertWeights(
            w1_cur.C, w2_cur.C, w3_cur.C,
            zeros(Float32, hidden_dim), zeros(Float32, hidden_dim),
            :cur, true,
            w1_cur.C, w1_cur.U, w1_cur.R  # Store CUR components
        )
    else
        # Standard or gated expert (same weight structure)
        return MoEExpertWeights(
            randn(Float32, dim, hidden_dim) .* σ,      # w1
            randn(Float32, hidden_dim, dim) .* σ,      # w2  
            randn(Float32, dim, hidden_dim) .* σ,      # w3
            zeros(Float32, hidden_dim),               # hb1
            zeros(Float32, hidden_dim),               # hb2
            expert_type, false,
            nothing, nothing, nothing                 # No CUR components
        )
    end
end

"""
    create_moe_run_state(config::MoELlamaConfig)

Create run state with all necessary buffers for MoE computation.
"""
function create_moe_run_state(config::MoELlamaConfig)
    # Create underlying Llama2 run state
    llama_state = Llama2.RunState(config.llama_config)
    
    # Create MoE-specific buffers
    num_experts = config.moe_num_experts
    top_k = config.moe_top_k
    dim = config.dim
    
    return MoERunState(
        llama_state,
        # MoE buffers
        zeros(Float32, num_experts),                    # router_logits
        zeros(Float32, top_k),                         # expert_gates
        zeros(Int, top_k),                             # selected_experts
        [zeros(Float32, dim) for _ in 1:num_experts],  # expert_outputs
        zeros(Float32, dim),                           # moe_temp_buffer
        # Tracking buffers  
        Float32[],                                     # auxiliary_loss_values
        Float32[],                                     # routing_entropy
        zeros(Int, num_experts),                       # expert_load_counts
        # Performance stats
        Dict{Symbol, Any}(
            :total_tokens => 0,
            :moe_layer_calls => 0,
            :expert_activations => 0,
            :routing_time => 0.0,
            :expert_compute_time => 0.0
        )
    )
end

"""
Utility functions for type checking and validation
"""

"""
    is_moe_layer(layer::MoETransformerLayerWeights)

Check if a layer uses MoE (vs dense FFN).
"""
is_moe_layer(layer::MoETransformerLayerWeights) = layer.use_moe

"""
    get_expert_count(layer::MoETransformerLayerWeights)

Get number of experts in a MoE layer (0 for dense layers).
"""
function get_expert_count(layer::MoETransformerLayerWeights)
    return layer.use_moe ? length(layer.moe_experts) : 0
end

"""
    get_moe_layer_indices(model::MoELanguageModel)

Get indices of all MoE layers in the model.
"""
function get_moe_layer_indices(model::MoELanguageModel)
    return [i for (i, layer) in enumerate(model.weights.layers) if is_moe_layer(layer)]
end

"""
    validate_matrix_dimensions(config::MoELlamaConfig)

Validate that all matrix dimensions are correct for Llama2 compatibility.
Critical for ensuring matmul! operations work correctly.
"""
function validate_matrix_dimensions(config::MoELlamaConfig)
    dim = config.dim
    hidden_dim = config.hidden_dim
    vocab_size = config.vocab_size
    num_experts = config.moe_num_experts
    
    # These dimension requirements are CRITICAL for Llama2.matmul! compatibility
    checks = [
        (true, "dim must be positive"),
        (hidden_dim > 0, "hidden_dim must be positive"), 
        (vocab_size > 0, "vocab_size must be positive"),
        (num_experts > 0, "num_experts must be positive"),
        (config.moe_top_k <= num_experts, "top_k must not exceed num_experts"),
        (dim % config.n_heads == 0, "dim must be divisible by n_heads"),
        # Router weight will be (dim, num_experts) for matmul!(logits, router, input)
        # Expert weights: w1(dim, hidden), w2(hidden, dim), w3(dim, hidden)
        # Output weight: (dim, vocab_size) for matmul!(logits, output, final_hidden)
    ]
    
    for (condition, message) in checks
        if !condition
            throw(ArgumentError("Invalid configuration: $message"))
        end
    end
    
    return true
end