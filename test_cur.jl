
using Llama2
using LinearAlgebra
using Statistics
using Printf

# Load MoE library
include("src/MixtureOfExperts.jl")
using .MixtureOfExperts
import .MixtureOfExperts: create_moe_run_state,convert_to_moe , MoELlamaConfig, TopKGating,SwitchTransformerLoss,convert_to_moe, GatingMechanism, LoadBalancingLoss, 
                         TopKGating, SwitchTransformerLoss, MoELanguageModel,moe_attention!,apply_rope!,MoEConfig,MoETransformerWeights,MoEKVCache,MoERunState,sample_moe,
                         MoELlamaConfig, count_parameters, count_active_parameters,compute_gates,create_moe_expert_weights,gated_expert_forward!,create_moe_layer,MoEExpertWeights,MoETransformerLayerWeights,moe_transformer!,reset_moe_state!
using Llama2
using LinearAlgebra
using Statistics
using Printf

# Load MoE library
include("src/MixtureOfExperts.jl")
using .MixtureOfExperts
import .MixtureOfExperts: convert_to_moe

function test_cur_integration_debug()
    println("🔧 Testing CUR integration with debugging...")
    
    # Load real model
    original_model = Llama2.load_karpathy_model("stories42M.bin", "tokenizer.bin")
    
    println("Original model config:")
    println("  dim: $(original_model.config.dim)")
    println("  hidden_dim: $(original_model.config.hidden_dim)")
    
    # Start with moderate compression
    cur_rank = original_model.config.hidden_dim ÷ 8  # More conservative rank
    println("Using CUR rank: $cur_rank")
    
    # Convert with CUR compression
    cur_model = convert_to_moe(
        original_model,
        [2];                       # Convert only layer 2 first
        num_experts=2,             # Fewer experts for debugging
        top_k=1,                   # Top-1 for simplicity
        expert_type=:cur,          # Enable CUR
        use_cur=true,
        cur_rank=cur_rank,         # Conservative compression
        expert_init_strategy=:perturb,
        cur_oversample=5
    )
    
    # Count parameters
    original_params = count_llama_parameters(original_model)
    cur_total = count_parameters(cur_model)
    cur_active = count_active_parameters(cur_model)
    
    println("\nParameter comparison:")
    println("  Original:     $(original_params)")
    println("  CUR Total:    $(cur_total)")
    println("  CUR Active:   $(cur_active)")
    println("  Compression:  $(round((1 - cur_total/Float64(original_params))*100, digits=1))%")
    
    # Test a simple forward pass first
    println("\nTesting single token forward pass...")
    state = create_moe_run_state(cur_model.config)
    
    try
        moe_transformer!(1, 1, cur_model, state)
        println("✅ Single token forward pass successful!")
        
        # Now test generation
        println("\nTesting CUR generation:")
        output = sample_moe(cur_model, "The brave knight"; 
                           temperature=0.0f0,
                           show_expert_stats=false,
                           max_seq_len=20,
                           verbose=false)
        
        println("Generated: \"$output\"")
        println("✅ CUR integration successful!")
        
    catch e
        println("❌ CUR forward pass failed: $e")
        println("Stack trace:")
        Base.showerror(stdout, e, catch_backtrace())
        return nothing
    end
    
    return cur_model
end

# Run the debug test
#cur_model = test_cur_integration_debug()
function debug_cur_decomposition()
    println("🔍 Debugging CUR decomposition...")
    
    # Create a test matrix matching your model dimensions
    test_matrix = randn(Float32, 512, 1376)  # w1 dimensions from your model
    println("Original matrix: $(size(test_matrix))")
    
    # Try CUR decomposition with different ranks
    for rank in [50, 100, 172]
        println("\n--- Testing rank $rank ---")
        try
            cur_decomp = cur_decompose(test_matrix; rank=rank, oversample=10)
            
            println("CUR dimensions:")
            println("  C: $(size(cur_decomp.C))")
            println("  U: $(size(cur_decomp.U))")  
            println("  R: $(size(cur_decomp.R))")
            
            # Test reconstruction
            reconstructed = cur_decomp.C * cur_decomp.U * cur_decomp.R
            println("  Reconstructed: $(size(reconstructed))")
            
            # Test transpose multiplication
            test_input = randn(Float32, 512)  # Input vector
            test_output = zeros(Float32, 1376)  # Output vector
            
            println("  Testing transpose mult: $(size(test_matrix))^T * $(size(test_input)) → $(size(test_output))")
            
            # Direct computation
            direct_result = test_matrix' * test_input
            println("  Direct result: $(size(direct_result))")
            
            # CUR computation 
            temp1 = cur_decomp.C' * test_input
            println("  C' * input: $(size(cur_decomp.C'))  * $(size(test_input)) → $(size(temp1))")
            
            temp2 = cur_decomp.U' * temp1  
            println("  U' * temp1: $(size(cur_decomp.U')) * $(size(temp1)) → $(size(temp2))")
            
            result = cur_decomp.R' * temp2
            println("  R' * temp2: $(size(cur_decomp.R')) * $(size(temp2)) → $(size(result))")
            
            error_norm = norm(direct_result - result)
            println("  Reconstruction error: $(error_norm)")
            
            # Count parameters
            original_params = length(test_matrix)
            cur_params = length(cur_decomp.C) + length(cur_decomp.U) + length(cur_decomp.R)
            compression = (1 - cur_params/original_params) * 100
            println("  Compression: $(round(compression, digits=1))%")
            
        catch e
            println("  ❌ Failed: $e")
        end
    end
end

# Run the debug
#debug_cur_decomposition()
function test_cur_integration_optimal()
    println("🔧 Testing CUR with optimal settings...")
    
    original_model = Llama2.load_karpathy_model("stories42M.bin", "tokenizer.bin")
    
    # Use the sweet spot rank from debug (100 had lowest error)
    cur_model = convert_to_moe(
        original_model,
        [2, 4, 6],                 # Convert 3 layers for substantial compression
        num_experts=4,
        top_k=2,
        expert_type=:cur,
        use_cur=true,
        cur_rank=100,              # Optimal rank from debug
        expert_init_strategy=:perturb,
        cur_oversample=10
    )
    
    # Count parameters
    original_params = count_llama_parameters(original_model)
    cur_total = count_parameters(cur_model)
    cur_active = count_active_parameters(cur_model)
    
    println("\nOptimal CUR Results:")
    println("  Original:     $(original_params)")
    println("  CUR Total:    $(cur_total)")  
    println("  CUR Active:   $(cur_active)")
    println("  Compression:  $(round((1 - cur_total/Float64(original_params))*100, digits=1))%")
    
    # Test generation
    println("\nTesting generation:")
    output = sample_moe(cur_model, "The brave knight"; 
                       temperature=0.0f0,
                       show_expert_stats=true,
                       max_seq_len=100)
    
    return cur_model
end

# Run with optimal settings
cur_model = test_cur_integration_optimal()