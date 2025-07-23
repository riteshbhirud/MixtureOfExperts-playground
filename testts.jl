#!/usr/bin/env julia

"""
Diagnose the type hierarchy issue with GatingMechanism and TopKGating
"""


println("🔍 Diagnosing type hierarchy issue...")

# Load the module
include("src/MixtureOfExperts.jl")
using .MixtureOfExperts

println("\n1️⃣ Checking if types exist:")

# Check GatingMechanism
try
    gm = MixtureOfExperts.GatingMechanism
    println("✅ GatingMechanism exists: $gm")
    println("   Type: $(typeof(gm))")
catch e
    println("❌ GatingMechanism error: $e")
end

# Check TopKGating
try
    # Try to create a TopKGating
    tkg = MixtureOfExperts.TopKGating(2)
    println("✅ TopKGating can be created: $tkg")
    println("   Type: $(typeof(tkg))")
    println("   Supertype: $(supertype(typeof(tkg)))")
    
    # Check the full type hierarchy
    current_type = typeof(tkg)
    println("   Type hierarchy:")
    while current_type != Any
        println("     $current_type")
        current_type = supertype(current_type)
    end
    
    global test_gate = tkg  # Save for further testing
    
catch e
    println("❌ TopKGating error: $e")
end

println("\n2️⃣ Testing type relationships:")

try
    gate = MixtureOfExperts.TopKGating(2)
    gm_type = MixtureOfExperts.GatingMechanism
    
    println("Gate type: $(typeof(gate))")
    println("GatingMechanism type: $gm_type")
    
    # Test isa relationship
    is_gating = isa(gate, gm_type)
    println("isa(gate, GatingMechanism): $is_gating")
    
    # Test subtype relationship
    is_subtype = typeof(gate) <: gm_type
    println("typeof(gate) <: GatingMechanism: $is_subtype")
    
    # Check what GatingMechanism actually is
    println("GatingMechanism nature: $(typeof(gm_type))")
    
catch e
    println("❌ Type relationship test error: $e")
end

println("\n3️⃣ Checking module structure:")

# Look at what's actually defined in the gating files
gating_files = [
    "src/gating/base.jl",
    "src/gating/topk.jl", 
    "src/gating/simple.jl"
]

for file in gating_files
    if isfile(file)
        println("📄 $file:")
        content = read(file, String)
        
        # Look for abstract type definition
        if occursin("abstract type GatingMechanism", content)
            println("   ✅ Defines GatingMechanism abstract type")
        end
        
        # Look for TopKGating struct
        if occursin("struct TopKGating", content)
            if occursin("<: GatingMechanism", content)
                println("   ✅ TopKGating inherits from GatingMechanism")
            else
                println("   ❌ TopKGating does NOT inherit from GatingMechanism")
            end
        end
        
        # Look for other issues
        if occursin("TopKGating", content) && !occursin("struct TopKGating", content)
            println("   ⚠️  Contains TopKGating references but no struct definition")
        end
        
    else
        println("❌ $file not found")
    end
end

println("\n4️⃣ Manual include test:")

try
    # Try including the files manually in correct order
    include("src/gating/base.jl")
    include("src/gating/topk.jl")
    
    # Now test again
    gate = TopKGating(2)
    println("✅ After manual include - TopKGating created: $(typeof(gate))")
    println("   Supertype: $(supertype(typeof(gate)))")
    println("   isa(gate, GatingMechanism): $(isa(gate, GatingMechanism))")
    
catch e
    println("❌ Manual include test failed: $e")
end

println("\n💡 Potential fixes:")
println("1. Check that TopKGating struct has '< GatingMechanism' in its definition")
println("2. Ensure gating/base.jl is included before gating/topk.jl")  
println("3. Check for any naming conflicts or duplicate definitions")
println("4. Try restarting Julia to clear any cached type definitions")