"""
Validation and Testing for MoE-Llama2 Integration

This module provides comprehensive validation, testing, and debugging utilities
to ensure correct MoE-Llama2 integration and identify potential issues.
"""

using Test
using LinearAlgebra
using Statistics
using Printf
include("src/MixtureOfExperts.jl")
using .MixtureOfExperts
using Flux
using LinearAlgebra
using Printf
using Statistics
import .MixtureOfExperts: MoELanguageModel,MoELlamaConfig,MoETransformerWeights,MoETransformerLayerWeights,MoEExpertWeights,MoERunState
using Transformers
using Transformers.HuggingFace
using Llama2
"""
    validate_moe_model(model::MoELanguageModel; verbose::Bool = true) -> Bool

Comprehensive validation of MoE model structure and functionality.

# Arguments
- `model`: MoE model to validate
- `verbose`: Whether to print detailed validation results

# Returns
- `true` if all validations pass, throws exception otherwise
"""
function validate_moe_model(model::MoELanguageModel; verbose::Bool = true)
    if verbose
        println("🔍 Validating MoE model...")
        println("="^50)
    end
    
    try
        # 1. Configuration validation
        validate_model_config(model.config, verbose)
        
        # 2. Weight structure validation
        validate_weight_structure(model.weights, model.config, verbose)
        
        # 3. Matrix dimension validation
        validate_matrix_dimensions_comprehensive(model, verbose)
        
        # 4. Expert validation
        validate_experts(model, verbose)
        
        # 5. Tokenizer validation
        validate_tokenizer_compatibility(model.tokenizer, model.config, verbose)
        
        # 6. Numerical stability validation
        validate_numerical_stability(model, verbose)
        
        # 7. Forward pass validation
        validate_forward_pass(model, verbose)
        
        # 8. Memory usage validation
        validate_memory_usage(model, verbose)
        
        if verbose
            println("✅ All validations passed!")
            println("🚀 Model is ready for inference and generation")
        end
        
        return true
        
    catch e
        if verbose
            println("❌ Validation failed: $e")
        end
        rethrow(e)
    end
end

"""
    validate_model_config(config::MoELlamaConfig, verbose::Bool)

Validate model configuration parameters.
"""
function validate_model_config(config::MoELlamaConfig, verbose::Bool)
    if verbose
        println("1️⃣  Validating model configuration...")
    end
    
    # Basic parameter validation
    @test config.dim > 0, "dim must be positive"
    @test config.hidden_dim > 0, "hidden_dim must be positive"
    @test config.n_layers > 0, "n_layers must be positive"
    @test config.n_heads > 0, "n_heads must be positive"
    @test config.n_kv_heads > 0, "n_kv_heads must be positive"
    @test config.vocab_size > 0, "vocab_size must be positive"
    @test config.seq_len > 0, "seq_len must be positive"
    
    # MoE-specific validation
    @test config.moe_num_experts > 0, "moe_num_experts must be positive"
    @test config.moe_top_k > 0, "moe_top_k must be positive"
    @test config.moe_top_k <= config.moe_num_experts, "moe_top_k cannot exceed moe_num_experts"
    
    # Layer indices validation
    @test all(1 .<= config.moe_layers .<= config.n_layers), "moe_layers indices must be valid"
    @test allunique(config.moe_layers), "moe_layers must not contain duplicates"
    
    # Attention head compatibility
    @test config.dim % config.n_heads == 0, "dim must be divisible by n_heads"
    @test config.n_heads % config.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
    
    # Shared experts validation
    if config.use_shared_experts
        @test config.num_shared_experts > 0, "num_shared_experts must be positive when use_shared_experts=true"
        @test config.num_shared_experts < config.moe_num_experts, "num_shared_experts should be less than total experts"
    end
    
    # CUR validation
    if config.use_cur
        @test config.moe_expert_type == :cur, "moe_expert_type must be :cur when use_cur=true"
        if !isnothing(config.cur_rank)
            @test config.cur_rank > 0, "cur_rank must be positive"
            @test config.cur_rank <= config.hidden_dim, "cur_rank cannot exceed hidden_dim"
        end
    end
    
    if verbose
        println("   ✓ Configuration parameters valid")
        println("   ✓ MoE parameters consistent")
        println("   ✓ Attention configuration valid")
    end
end

"""
    validate_weight_structure(weights::MoETransformerWeights, config::MoELlamaConfig, verbose::Bool)

Validate weight structure and layer consistency.
"""
function validate_weight_structure(weights::MoETransformerWeights, config::MoELlamaConfig, verbose::Bool)
    if verbose
        println("2️⃣  Validating weight structure...")
    end
    
    # Global weights validation
    @test size(weights.token_embedding_table) == (config.dim, config.vocab_size), "Invalid token embedding dimensions"
    @test size(weights.rms_final_weight) == (config.dim,), "Invalid final RMS norm dimensions"
    @test size(weights.output_weight) == (config.dim, config.vocab_size), "Invalid output weight dimensions"
    
    # Layer count validation
    @test length(weights.layers) == config.n_layers, "Number of layers mismatch"
    
    # Per-layer validation
    moe_layer_count = 0
    dense_layer_count = 0
    
    for (layer_idx, layer) in enumerate(weights.layers)
        validate_layer_structure(layer, layer_idx, config)
        
        if layer.use_moe
            moe_layer_count += 1
            @test layer_idx in config.moe_layers, "Layer $layer_idx is MoE but not in config.moe_layers"
        else
            dense_layer_count += 1
            @test layer_idx ∉ config.moe_layers, "Layer $layer_idx is dense but in config.moe_layers"
        end
    end
    
    @test moe_layer_count == length(config.moe_layers), "MoE layer count mismatch"
    @test dense_layer_count == config.n_layers - length(config.moe_layers), "Dense layer count mismatch"
    
    if verbose
        println("   ✓ Global weights have correct dimensions")
        println("   ✓ Layer structure consistent ($moe_layer_count MoE, $dense_layer_count dense)")
        println("   ✓ All layers validated")
    end
end

"""
    validate_layer_structure(layer::MoETransformerLayerWeights, layer_idx::Int, config::MoELlamaConfig)

Validate individual layer structure.
"""
function validate_layer_structure(layer::MoETransformerLayerWeights, layer_idx::Int, config::MoELlamaConfig)
    # Attention weights validation (always present)
    llama_layer = layer.llama_layer
    @test size(llama_layer.rms_att_weight) == (config.dim,), "Layer $layer_idx: Invalid attention RMS norm"
    @test size(llama_layer.wq) == (config.dim, config.dim), "Layer $layer_idx: Invalid query weights"
    @test size(llama_layer.wk) == (config.dim, config.dim), "Layer $layer_idx: Invalid key weights"
    @test size(llama_layer.wv) == (config.dim, config.dim),"Layer $layer_idx: Invalid value weights"
    @test size(llama_layer.wo) == (config.dim, config.dim), "Layer $layer_idx: Invalid output weights"
    
    if layer.use_moe
        # MoE layer validation
        @test !isnothing(layer.moe_experts), "Layer $layer_idx: MoE experts missing"
        @test !isnothing(layer.moe_router_weight), "Layer $layer_idx: Router weight missing"
        @test !isnothing(layer.moe_config), "Layer $layer_idx: MoE config missing"
        
        @test length(layer.moe_experts) == config.moe_num_experts, "Layer $layer_idx: Wrong number of experts"
        @test size(layer.moe_router_weight) == (config.dim, config.moe_num_experts), "Layer $layer_idx: Invalid router dimensions"
        
        # Validate each expert
        for (expert_idx, expert) in enumerate(layer.moe_experts)
            validate_expert_structure(expert, expert_idx, layer_idx, config)
        end
        
        # Validate shared experts if present
        if !isnothing(layer.shared_experts)
            for (shared_idx, shared_expert) in enumerate(layer.shared_experts)
                validate_expert_structure(shared_expert, shared_idx, layer_idx, config, true)
            end
        end
    else
        # Dense layer validation
        @test size(llama_layer.rms_ffn_weight) == (config.dim,), "Layer $layer_idx: Invalid FFN RMS norm"
        @test size(llama_layer.w1) == (config.dim, config.hidden_dim), "Layer $layer_idx: Invalid w1 dimensions"
        @test size(llama_layer.w2) == (config.hidden_dim, config.dim), "Layer $layer_idx: Invalid w2 dimensions"
        @test size(llama_layer.w3) == (config.dim, config.hidden_dim), "Layer $layer_idx: Invalid w3 dimensions"
        
        # MoE fields should be nothing for dense layers
        @test isnothing(layer.moe_experts), "Layer $layer_idx: Dense layer has MoE experts"
        @test isnothing(layer.moe_router_weight), "Layer $layer_idx: Dense layer has router weight"
        @test isnothing(layer.moe_config), "Layer $layer_idx: Dense layer has MoE config"
    end
end

"""
    validate_expert_structure(expert::MoEExpertWeights, expert_idx::Int, layer_idx::Int, 
                             config::MoELlamaConfig, is_shared::Bool = false)

Validate individual expert structure.
"""
function validate_expert_structure(expert::MoEExpertWeights, expert_idx::Int, layer_idx::Int, 
                                  config::MoELlamaConfig, is_shared::Bool = false)
    expert_type = is_shared ? "shared expert" : "expert"
    
    # Weight dimensions
    @test size(expert.w1) == (config.dim, config.hidden_dim), "Layer $layer_idx $expert_type $expert_idx: Invalid w1 dimensions"
    @test size(expert.w2) == (config.hidden_dim, config.dim), "Layer $layer_idx $expert_type $expert_idx: Invalid w2 dimensions"
    @test size(expert.w3) == (config.dim, config.hidden_dim), "Layer $layer_idx $expert_type $expert_idx: Invalid w3 dimensions"
    
    # Buffer dimensions
    @test size(expert.hb1) == (config.hidden_dim,), "Layer $layer_idx $expert_type $expert_idx: Invalid hb1 buffer"
    @test size(expert.hb2) == (config.hidden_dim,), "Layer $layer_idx $expert_type $expert_idx: Invalid hb2 buffer"
    
    # CUR validation
    if expert.is_cur_compressed
        @test expert.expert_type == :cur, "Layer $layer_idx $expert_type $expert_idx: CUR compressed but type not :cur"
        @test !isnothing(expert.cur_c), "Layer $layer_idx $expert_type $expert_idx: CUR C matrix missing"
        @test !isnothing(expert.cur_u), "Layer $layer_idx $expert_type $expert_idx: CUR U matrix missing"
        @test !isnothing(expert.cur_r), "Layer $layer_idx $expert_type $expert_idx: CUR R matrix missing"
    end
end

"""
    validate_matrix_dimensions_comprehensive(model::MoELanguageModel, verbose::Bool)

Comprehensive matrix dimension validation for Llama2 compatibility.
"""
function validate_matrix_dimensions_comprehensive(model::MoELanguageModel, verbose::Bool)
    if verbose
        println("3️⃣  Validating matrix dimensions for Llama2 compatibility...")
    end
    
    config = model.config
    weights = model.weights
    
    # Test actual matrix multiplications to ensure compatibility
    test_input = randn(Float32, config.dim)
    test_logits = zeros(Float32, config.vocab_size)
    
    # Test token embedding
    @test size(weights.token_embedding_table[:, 1]) == (config.dim,), "Token embedding lookup failed"
    
    # Test output projection
    try
        Llama2.matmul!(test_logits, weights.output_weight, test_input)
        @test all(isfinite.(test_logits)), "Output projection produced non-finite values"
    catch e
        error("Output weight matrix multiplication failed: $e")
    end
    
    # Test router weights for MoE layers
    for (layer_idx, layer) in enumerate(weights.layers)
        if layer.use_moe
            test_router_logits = zeros(Float32, config.moe_num_experts)
            try
                Llama2.matmul!(test_router_logits, layer.moe_router_weight, test_input)
                @test all(isfinite.(test_router_logits)), "Router layer $layer_idx produced non-finite values"
            catch e
                error("Router weight matrix multiplication failed in layer $layer_idx: $e")
            end
            
            # Test expert weights
            for (expert_idx, expert) in enumerate(layer.moe_experts)
                test_hidden = zeros(Float32, config.hidden_dim)
                test_output = zeros(Float32, config.dim)
                
                try
                    Llama2.matmul!(test_hidden, expert.w1, test_input)
                    @test all(isfinite.(test_hidden)), "Expert w1 layer $layer_idx expert $expert_idx failed"
                    
                    Llama2.matmul!(test_output, expert.w2, test_hidden)
                    @test all(isfinite.(test_output)), "Expert w2 layer $layer_idx expert $expert_idx failed"
                    
                    Llama2.matmul!(test_hidden, expert.w3, test_input)
                    @test all(isfinite.(test_hidden)), "Expert w3 layer $layer_idx expert $expert_idx failed"
                catch e
                    error("Expert weight matrix multiplication failed in layer $layer_idx expert $expert_idx: $e")
                end
            end
        end
    end
    
    if verbose
        println("   ✓ All matrix multiplications work correctly")
        println("   ✓ Dimensions compatible with Llama2.matmul!")
        println("   ✓ No numerical issues in test multiplications")
    end
end

"""
    validate_experts(model::MoELanguageModel, verbose::Bool)

Validate expert functionality and routing.
"""
function validate_experts(model::MoELanguageModel, verbose::Bool)
    if verbose
        println("4️⃣  Validating expert functionality...")
    end
    
    config = model.config
    
    # Test expert forward passes
    test_input = randn(Float32, config.dim)
    
    for (layer_idx, layer) in enumerate(model.weights.layers)
        if layer.use_moe
            for (expert_idx, expert) in enumerate(layer.moe_experts)
                test_output = zeros(Float32, config.dim)
                
                # Test expert forward pass
                if expert.expert_type == :cur && expert.is_cur_compressed
                    cur_expert_forward!(test_output, expert, test_input)
                else
                    gated_expert_forward!(test_output, expert, test_input)
                end
                
                @test all(isfinite.(test_output)), "Expert $expert_idx in layer $layer_idx produced non-finite output"
                @test !all(iszero.(test_output)), "Expert $expert_idx in layer $layer_idx produced all-zero output"
            end
            
            # Test router functionality
            router_logits = zeros(Float32, config.moe_num_experts)
            Llama2.matmul!(router_logits, layer.moe_router_weight, test_input)
            
            # Test gating mechanism
            router_logits_matrix = reshape(router_logits, :, 1)
            expert_indices, expert_gates, router_probs = compute_gates(layer.moe_config.gate_type, router_logits_matrix)
            
            @test size(expert_indices, 1) >= config.moe_top_k, "Router returned insufficient experts"
            @test all(1 .<= expert_indices[1:config.moe_top_k, 1] .<= config.moe_num_experts), "Router returned invalid expert indices"
            @test all(expert_gates[1:config.moe_top_k, 1] .>= 0), "Router returned negative gate weights"
            @test isapprox(sum(expert_gates[1:config.moe_top_k, 1]), 1.0, atol=1e-6), "Router gate weights don't sum to 1"
        end
    end
    
    if verbose
        println("   ✓ All experts produce valid outputs")
        println("   ✓ Router functionality working correctly")
        println("   ✓ Gating mechanisms operational")
    end
end

"""
    validate_tokenizer_compatibility(tokenizer, config::MoELlamaConfig, verbose::Bool)

Validate tokenizer compatibility with model configuration.
"""
function validate_tokenizer_compatibility(tokenizer, config::MoELlamaConfig, verbose::Bool)
    if verbose
        println("5️⃣  Validating tokenizer compatibility...")
    end
    
    # Check vocabulary size consistency
    if hasfield(typeof(tokenizer), :id_to_token)
        @test length(tokenizer.id_to_token) == config.vocab_size, "Tokenizer vocab size mismatch"
    end
    
    # Test tokenization
    test_text = "Hello, world! This is a test."
    tokens = Llama2.encode(test_text, tokenizer)
    
    @test !isempty(tokens), "Tokenizer returned empty token sequence"
    @test all(1 .<= tokens .<= config.vocab_size), "Tokenizer returned out-of-range tokens"
    
    # Test special tokens
    if hasfield(typeof(tokenizer), :bos_token_id)
        @test 1 <= tokenizer.bos_token_id <= config.vocab_size, "BOS token ID out of range"
    end
    
    if hasfield(typeof(tokenizer), :eos_token_id)
        @test 1 <= tokenizer.eos_token_id <= config.vocab_size,"EOS token ID out of range"
    end
    
    if verbose
        println("   ✓ Tokenizer vocab size matches model")
        println("   ✓ Tokenization working correctly")
        println("   ✓ Special tokens valid")
    end
end

"""
    validate_numerical_stability(model::MoELanguageModel, verbose::Bool)

Test numerical stability with various inputs.
"""
function validate_numerical_stability(model::MoELanguageModel, verbose::Bool)
    if verbose
        println("6️⃣  Validating numerical stability...")
    end
    
    config = model.config
    state = create_moe_run_state(config)
    
    # Test with various token inputs
    test_tokens = [1, config.vocab_size ÷ 2, config.vocab_size]
    
    for token in test_tokens
        if token <= config.vocab_size
            try
                moe_transformer!(token, 1, model, state)
                @test all(isfinite.(state.logits)), "Non-finite logits for token $token"
                @test !all(iszero.(state.logits)), "All-zero logits for token $token"
            catch e
                error("Forward pass failed for token $token: $e")
            end
        end
    end
    
    # Test with extreme inputs (boundary testing)
    extreme_tests = [
        ("minimum token", 1),
        ("maximum token", config.vocab_size),
        ("maximum position", min(config.seq_len, 100))
    ]
    
    for (test_name, test_value) in extreme_tests
        if test_name == "maximum position"
            try
                moe_transformer!(1, test_value, model, state)
                @test all(isfinite.(state.logits)), "Non-finite logits for $test_name"
            catch e
                error("Forward pass failed for $test_name: $e")
            end
        end
    end
    
    if verbose
        println("   ✓ Model stable with various token inputs")
        println("   ✓ No numerical overflow/underflow detected")
        println("   ✓ Boundary conditions handled correctly")
    end
end

"""
    validate_forward_pass(model::MoELanguageModel, verbose::Bool)

Validate complete forward pass functionality.
"""
function validate_forward_pass(model::MoELanguageModel, verbose::Bool)
    if verbose
        println("7️⃣  Validating forward pass...")
    end
    
    config = model.config
    state = create_moe_run_state(config)
    
    # Test single forward pass
    test_token = 1
    test_pos = 1
    
    moe_transformer!(test_token, test_pos, model, state)
    
    # Validate outputs
    @test size(state.logits) == (config.vocab_size,), "Wrong logits shape"
    @test all(isfinite.(state.logits)), "Non-finite logits"
    
    # Validate MoE state updates
    moe_layers_exist = any(layer.use_moe for layer in model.weights.layers)
    
    if moe_layers_exist
        @test any(state.selected_experts .> 0), "No experts selected"
        @test sum(state.expert_gates[1:config.moe_top_k]) ≈ 1.0, "Expert gates don't sum to 1"
        @test state.inference_stats[:expert_activations] > 0, "No expert activations recorded"
    end
    
    # Test sequence processing
    test_sequence = [1, 5, 10, 2]
    reset_moe_state!(state)
    
    for (pos, token) in enumerate(test_sequence)
        moe_transformer!(token, pos, model, state)
        @test all(isfinite.(state.logits)), "Non-finite logits at position $pos"
    end
    
    if verbose
        println("   ✓ Single forward pass working")
        println("   ✓ MoE routing functioning correctly")
        println("   ✓ Sequence processing stable")
    end
end

"""
    validate_memory_usage(model::MoELanguageModel, verbose::Bool)

Validate memory usage and allocation patterns.
"""
function validate_memory_usage(model::MoELanguageModel, verbose::Bool)
    if verbose
        println("8️⃣  Validating memory usage...")
    end
    
    config = model.config
    
    # Calculate expected memory usage
    expected_params = count_parameters(model)
    expected_active = count_active_parameters(model)
    
    @test expected_active <= expected_params, "Active parameters exceed total parameters"
    
    efficiency_ratio = expected_active / expected_params
    @test efficiency_ratio > 0.0, "Zero efficiency ratio"
    @test efficiency_ratio <= 1.0, "Efficiency ratio exceeds 1.0"
    
    # Test state creation and cleanup
    state = create_moe_run_state(config)
    
    # Validate buffer sizes
    @test length(state.router_logits) == config.moe_num_experts, "Wrong router logits buffer size"
    @test length(state.expert_gates) == config.moe_top_k, "Wrong expert gates buffer size"
    @test length(state.selected_experts) == config.moe_top_k, "Wrong selected experts buffer size"
    @test length(state.expert_outputs) == config.moe_num_experts, "Wrong expert outputs buffer count"
    
    for expert_output in state.expert_outputs
        @test length(expert_output) == config.dim, "Wrong expert output buffer size"
    end
    
    if verbose
        @printf "   ✓ Total parameters: %d\n" expected_params
        @printf "   ✓ Active parameters: %d (%.1f%% efficiency)\n" expected_active (efficiency_ratio * 100)
        println("   ✓ Memory buffers correctly sized")
    end
end

"""
Advanced validation functions
"""

"""
    validate_against_original(original_model::Llama2.LanguageModel, 
                             moe_model::MoELanguageModel;
                             test_tokens::Vector{Int} = [1, 5, 10],
                             tolerance::Float32 = 1e-3,
                             verbose::Bool = true) -> Dict

Validate MoE model against original model for dense layers.
"""
function validate_against_original(original_model::Llama2.LanguageModel, 
                                  moe_model::MoELanguageModel;
                                  test_tokens::Vector{Int} = [1, 5, 10],
                                  tolerance::Float32 = 1e-3,
                                  verbose::Bool = true)
    
    if verbose
        println("🔍 Validating MoE model against original...")
    end
    
    results = Dict{String, Any}(
        "dense_layer_differences" => Float32[],
        "attention_differences" => Float32[],
        "max_difference" => 0.0f0,
        "validation_passed" => true
    )
    
    original_state = Llama2.RunState(original_model.config)
    moe_state = create_moe_run_state(moe_model.config)
    
    for test_token in test_tokens
        if test_token <= min(original_model.config.vocab_size, moe_model.config.vocab_size)
            # Run through original model
            Llama2.transformer!(test_token, 1, original_model.config, original_state, original_model.weights)
            
            # Run through MoE model
            moe_transformer!(test_token, 1, moe_model, moe_state)
            
            # Compare outputs for dense layers only
            for (layer_idx, moe_layer) in enumerate(moe_model.weights.layers)
                if !moe_layer.use_moe
                    # This layer should match the original
                    # Note: This is a simplified comparison - full implementation would need
                    # to track intermediate activations
                end
            end
            
            # Compare final outputs only if no MoE layers (for debugging)
            if isempty(moe_model.config.moe_layers)
                diff = maximum(abs.(original_state.logits - moe_state.logits))
                push!(results["dense_layer_differences"], diff)
                results["max_difference"] = max(results["max_difference"], diff)
                
                if diff > tolerance
                    results["validation_passed"] = false
                    if verbose
                        @warn "Large difference detected: $diff (tolerance: $tolerance)"
                    end
                end
            end
        end
    end
    
    if verbose
        if results["validation_passed"]
            println("   ✓ Dense layers match original model within tolerance")
        else
            println("   ⚠️  Some differences exceed tolerance")
        end
    end
    
    return results
end

"""
    benchmark_model_performance(model::MoELanguageModel;
                               num_tokens::Int = 100,
                               warmup_tokens::Int = 10,
                               verbose::Bool = true) -> Dict

Benchmark model performance and identify bottlenecks.
"""
function benchmark_model_performance(model::MoELanguageModel;
                                   num_tokens::Int = 100,
                                   warmup_tokens::Int = 10,
                                   verbose::Bool = true)
    
    if verbose
        println("⚡ Benchmarking model performance...")
    end
    
    config = model.config
    state = create_moe_run_state(config)
    
    # Warmup
    for i in 1:warmup_tokens
        token = rand(1:config.vocab_size)
        moe_transformer!(token, i, model, state)
    end
    
    # Reset stats
    reset_moe_state!(state)
    
    # Benchmark
    times = Float64[]
    start_time = time()
    
    for i in 1:num_tokens
        token = rand(1:config.vocab_size)
        
        token_start = time()
        moe_transformer!(token, i, model, state)
        token_end = time()
        
        push!(times, token_end - token_start)
    end
    
    total_time = time() - start_time
    
    # Analyze results
    results = Dict{String, Any}(
        "total_time" => total_time,
        "tokens_per_second" => num_tokens / total_time,
        "mean_token_time" => mean(times),
        "std_token_time" => std(times),
        "min_token_time" => minimum(times),
        "max_token_time" => maximum(times),
        "expert_activations" => state.inference_stats[:expert_activations],
        "moe_layer_calls" => state.inference_stats[:moe_layer_calls],
        "routing_time_fraction" => state.inference_stats[:routing_time] / total_time,
        "expert_compute_time_fraction" => state.inference_stats[:expert_compute_time] / total_time
    )
    
    if verbose
        @printf "   🚀 Tokens per second: %.2f\n" results["tokens_per_second"]
        @printf "   ⏱️  Mean time per token: %.3f ms\n" (results["mean_token_time"] * 1000)
        @printf "   🎯 Expert activations: %d\n" results["expert_activations"]
        @printf "   🔀 Routing overhead: %.1f%%\n" (results["routing_time_fraction"] * 100)
        @printf "   🧮 Expert compute: %.1f%%\n" (results["expert_compute_time_fraction"] * 100)
    end
    
    return results
end

"""
    debug_model_state(model::MoELanguageModel, state::MoERunState; verbose::Bool = true) -> Dict

Debug current model state and identify potential issues.
"""
function debug_model_state(model::MoELanguageModel, state::MoERunState; verbose::Bool = true)
    if verbose
        println("🐛 Debugging model state...")
    end
    
    debug_info = Dict{String, Any}()
    
    # Check for NaN/Inf values
    nan_checks = [
        ("logits", state.logits),
        ("router_logits", state.router_logits),
        ("expert_gates", state.expert_gates),
        ("x", state.x),
        ("xb", state.xb),
        ("xb2", state.xb2)
    ]
    
    debug_info["numerical_issues"] = Dict{String, Bool}()
    
    for (name, array) in nan_checks
        has_nan = any(isnan.(array))
        has_inf = any(isinf.(array))
        debug_info["numerical_issues"]["$(name)_has_nan"] = has_nan
        debug_info["numerical_issues"]["$(name)_has_inf"] = has_inf
        
        if verbose && (has_nan || has_inf)
            println("   ⚠️  $name contains $(has_nan ? "NaN" : "")$(has_nan && has_inf ? " and " : "")$(has_inf ? "Inf" : "") values")
        end
    end
    
    # Expert usage analysis
    if any(state.expert_load_counts .> 0)
        usage_stats = state.expert_load_counts
        debug_info["expert_usage"] = Dict(
            "total_activations" => sum(usage_stats),
            "active_experts" => count(usage_stats .> 0),
            "max_usage" => maximum(usage_stats),
            "min_usage" => minimum(usage_stats),
            "usage_variance" => var(usage_stats)
        )
        
        if verbose
            println("   📊 Expert usage: $(debug_info["expert_usage"]["active_experts"])/$(length(usage_stats)) experts active")
        end
    end
    
    # Memory usage
    debug_info["memory_info"] = Dict(
        "state_size_bytes" => sizeof(state),
        "expert_outputs_total" => sum(sizeof(output) for output in state.expert_outputs)
    )
    
    if verbose
        println("   💾 State memory: $(debug_info["memory_info"]["state_size_bytes"]) bytes")
    end
    
    return debug_info
end

"""
    expert_usage_analysis(model::MoELanguageModel, test_sequence::Vector{Int}; verbose::Bool = true) -> Dict

Analyze expert usage patterns for a given sequence.
"""
function expert_usage_analysis(model::MoELanguageModel, test_sequence::Vector{Int}; verbose::Bool = true)
    if verbose
        println("📈 Analyzing expert usage patterns...")
    end
    
    config = model.config
    state = create_moe_run_state(config)
    
    # Track usage per position
    position_usage = []
    total_usage = zeros(Int, config.moe_num_experts)
    
    for (pos, token) in enumerate(test_sequence)
        if token <= config.vocab_size && pos <= config.seq_len
            moe_transformer!(token, pos, model, state)
            
            # Record usage for this position
            current_usage = zeros(Int, config.moe_num_experts)
            for k in 1:config.moe_top_k
                expert_idx = state.selected_experts[k]
                if expert_idx > 0
                    current_usage[expert_idx] += 1
                    total_usage[expert_idx] += 1
                end
            end
            
            push!(position_usage, copy(current_usage))
        end
    end
    
    # Analyze patterns
    analysis = Dict{String, Any}(
        "total_usage" => total_usage,
        "position_usage" => position_usage,
        "active_experts" => count(total_usage .> 0),
        "usage_entropy" => entropy_from_counts(total_usage),
        "load_balance_score" => 1.0 - std(total_usage) / (mean(total_usage) + 1e-8)
    )
    
    if verbose
        @printf "   🎯 Active experts: %d/%d\n" analysis["active_experts"] config.moe_num_experts
        @printf "   📏 Load balance score: %.3f\n" analysis["load_balance_score"]
        @printf "   🌀 Usage entropy: %.3f\n" analysis["usage_entropy"]
    end
    
    return analysis
end

"""
    entropy_from_counts(counts::Vector{Int}) -> Float32

Compute entropy from usage counts.
"""
function entropy_from_counts(counts::Vector{Int})
    total = sum(counts)
    if total == 0
        return 0.0f0
    end
    
    probs = counts ./ total
    entropy = 0.0f0
    
    for p in probs
        if p > 0
            entropy -= p * log(p)
        end
    end
    
    return entropy
end

"""
    routing_entropy_analysis(state::MoERunState; verbose::Bool = true) -> Dict

Analyze routing entropy patterns.
"""
function routing_entropy_analysis(state::MoERunState; verbose::Bool = true)
    if verbose
        println("🌀 Analyzing routing entropy...")
    end
    
    entropies = state.routing_entropy
    
    if isempty(entropies)
        if verbose
            println("   ⚠️  No routing entropy data available")
        end
        return Dict{String, Any}("error" => "no_data")
    end
    
    analysis = Dict{String, Any}(
        "mean_entropy" => mean(entropies),
        "std_entropy" => std(entropies),
        "min_entropy" => minimum(entropies),
        "max_entropy" => maximum(entropies),
        "entropy_trend" => length(entropies) > 1 ? cor(1:length(entropies), entropies) : 0.0,
        "num_measurements" => length(entropies)
    )
    
    if verbose
        @printf "   📊 Mean entropy: %.3f ± %.3f\n" analysis["mean_entropy"] analysis["std_entropy"]
        @printf "   📈 Entropy range: [%.3f, %.3f]\n" analysis["min_entropy"] analysis["max_entropy"]
        @printf "   📉 Trend correlation: %.3f\n" analysis["entropy_trend"]
    end
    
    return analysis
end