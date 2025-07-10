# Complete example showing how to replace dense FFN layers with MoE
# Following the mentor's instructions step by step

using Flux
using Random
using LinearAlgebra

# Include our MoE implementation
include("../src/MixtureOfExperts.jl")
using .MixtureOfExperts

# Include integrations
include("../src/integrations/llama2_moe_integration.jl")
include("../src/integrations/transformers_moe_integration.jl")

println("=== Step 1: Understanding Standard Dense FFN ===")

# This is what a standard Llama FFN looks like (what we're replacing)
struct StandardLlamaFFN
    w1::Dense  # gate projection  
    w2::Dense  # down projection
    w3::Dense  # up projection
end

function (ffn::StandardLlamaFFN)(x)
    gate = silu.(ffn.w1(x))  # Gate: w1(x) with SiLU activation
    up = ffn.w3(x)           # Up: w3(x)  
    hidden = gate .* up       # Element-wise multiplication
    return ffn.w2(hidden)     # Down: w2(gate * up)
end

# Example standard FFN
standard_ffn = StandardLlamaFFN(
    Dense(512, 2048),   # w1: input_dim -> hidden_dim
    Dense(2048, 512),   # w2: hidden_dim -> output_dim  
    Dense(512, 2048)    # w3: input_dim -> hidden_dim
)

println("Standard FFN parameters: ", sum([length(p) for p in Flux.params(standard_ffn)]))

println("\n=== Step 2: Simple MoE Replacement (Random Gating) ===")

# Step 2a: Create simple MoE config as mentor requested
# "at the beginning, just choose random expert"
simple_moe_config = MoEConfig(
    num_experts = 8,                    # 8 smaller experts instead of 1 big FFN
    expert_type = :gated,               # Use Llama-style gated experts
    input_dim = 512,
    hidden_dim = 2048 ÷ 8,             # Distribute parameters: 2048/8 = 256 per expert
    output_dim = 512,
    activation = silu,                  # SiLU activation like Llama
    gate_type = RandomGating(2),        # Random selection, top-2 as mentor requested
    top_k = 2,
    balance_loss = NoBalancingLoss(),   # No balancing loss for initial testing
    expert_dropout = 0.0f0
)

# Step 2b: Create the MoE layer
simple_moe = MoELayer(simple_moe_config)

println("MoE parameters: ", sum([length(p) for p in Flux.params(simple_moe)]))
println("Parameter reduction: ", round((1 - sum([length(p) for p in Flux.params(simple_moe)]) / 
                                        sum([length(p) for p in Flux.params(standard_ffn)])) * 100, digits=2), "%")

println("\n=== Step 3: Testing the Replacement ===")

# Test input
batch_size = 32
seq_len = 128
input_dim = 512

test_input = randn(Float32, input_dim, batch_size)

# Test standard FFN
println("Testing standard FFN...")
standard_output = standard_ffn(test_input)
println("Standard FFN output shape: ", size(standard_output))

# Test MoE replacement  
println("Testing MoE replacement...")
moe_output, moe_loss = simple_moe(test_input; training=true)
println("MoE output shape: ", size(moe_output))
println("MoE balance loss: ", moe_loss)

println("\n=== Step 4: Advanced MoE with Top-K Gating ===")

# Now add proper gating as shown in Stanford CS336
advanced_moe_config = MoEConfig(
    num_experts = 8,
    expert_type = :gated,
    input_dim = 512,
    hidden_dim = 256,  # 2048 ÷ 8
    output_dim = 512,
    activation = silu,
    gate_type = TopKGating(2),                    # Stanford CS336 Top-K gating
    balance_loss = SwitchTransformerLoss(0.01f0), # Stanford CS336 Switch loss
    use_fp32_router = true,                       # Numerical stability
    noise_scale = 0.01f0                          # Training noise
)

advanced_moe = MoELayer(advanced_moe_config)

println("Testing advanced MoE with Top-K gating...")
advanced_output, advanced_loss = advanced_moe(test_input; training=true)
println("Advanced MoE output shape: ", size(advanced_output))
println("Advanced MoE balance loss: ", advanced_loss)

println("\n=== Step 5: Integration with Actual Transformer ===")

# Simulate a minimal transformer layer structure
struct SimpleTransformerLayer
    attention_norm::LayerNorm
    attention::Dense  # Simplified attention
    ffn_norm::LayerNorm
    ffn::Union{StandardLlamaFFN, MoELayer}  # FFN or MoE
end

function (layer::SimpleTransformerLayer)(x; training::Bool = false)
    # Attention block (simplified)
    norm_x = layer.attention_norm(x)
    attn_out = layer.attention(norm_x)
    x = x + attn_out  # Residual
    
    # FFN block (this is what we replace)
    norm_x = layer.ffn_norm(x)
    
    if layer.ffn isa MoELayer
        ffn_out, loss = layer.ffn(norm_x; training=training)
        return x + ffn_out, loss
    else
        ffn_out = layer.ffn(norm_x)
        return x + ffn_out, 0.0f0
    end
end

# Create layers
println("Creating transformer layers...")

# Standard transformer layer
standard_layer = SimpleTransformerLayer(
    LayerNorm(512),
    Dense(512, 512),
    LayerNorm(512), 
    standard_ffn
)

# MoE transformer layer
moe_layer = SimpleTransformerLayer(
    LayerNorm(512),
    Dense(512, 512),
    LayerNorm(512),
    simple_moe
)

# Test both
test_sequence = randn(Float32, 512, seq_len, batch_size)

println("Testing standard transformer layer...")
standard_layer_output, _ = standard_layer(test_sequence[:, 1, 1:1]; training=true)
println("Standard layer output shape: ", size(standard_layer_output))

println("Testing MoE transformer layer...")
moe_layer_output, moe_layer_loss = moe_layer(test_sequence[:, 1, 1:1]; training=true)
println("MoE layer output shape: ", size(moe_layer_output))
println("MoE layer loss: ", moe_layer_loss)

println("\n=== Step 6: Analysis and Statistics ===")

# Print expert usage statistics
if haskey(simple_moe.training_stats, :tokens_per_expert)
    expert_usage = simple_moe.training_stats[:tokens_per_expert]
    println("Expert usage (random gating): ", expert_usage)
    println("Load balance score: ", load_balance_score(expert_usage))
end

println("\n=== Step 7: Next Steps (Following Mentor's Roadmap) ===")
println("✅ Step 1: Located dense layers in transformer (FFN blocks)")
println("✅ Step 2: Created MoE replacement with multiple smaller experts")  
println("✅ Step 3: Implemented random expert selection as requested")
println("✅ Step 4: Added proper Top-K gating and softmax routing")
println("✅ Step 5: Created integration points for Llama and Transformers.jl")
println()
println("🚀 Ready for next phase:")
println("   - Integrate with full Llama2.jl model")
println("   - Test on actual language modeling tasks") 
println("   - Add more sophisticated gating mechanisms")
println("   - Eventually integrate with Dagger.jl for dynamic scheduling")

println("\n=== Example Usage in Practice ===")

# Show how to use with actual model configs
println("For Llama2.jl integration:")
println("```julia")
println("# Replace this in transformer forward pass:")
println("# matmul!(s.hb, w.w1, s.xb)")  
println("# matmul!(s.hb2, w.w3, s.xb)")
println("# s.hb .*= s.hb2") 
println("# matmul!(s.xb, w.w2, s.hb)")
println("#")
println("# With this:")
println("# input_matrix = reshape(s.xb, :, 1)")
println("# moe_output, moe_loss = moe_layer(input_matrix; training=training)")
println("# s.xb = vec(moe_output)")
println("```")
println()
println("For Transformers.jl integration:")
println("```julia") 
println("# Replace Layers.Chain(LLamaGated(...), Layers.Dense(...)) with:")
println("# LlamaMoEBlock(moe_config)")
println("```")