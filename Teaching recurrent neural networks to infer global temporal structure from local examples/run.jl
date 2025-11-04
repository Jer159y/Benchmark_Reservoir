# Pkg 환경을 활성화 (프로젝트 폴더에 Project.toml이 있다고 가정)
using Pkg
Pkg.activate(".")

# 데이터 경로 추가 (필요시)
push!(LOAD_PATH, "../data")

println("Running figure generation scripts...")

# 각 스크립트를 include하여 실행
try
    include("fig_intro.jl")
    include("fig_translate_transform.jl")
    include("fig_bifurcate_period_double.jl")
    include("fig_differential.jl")
    include("fig_tanh_SN_SP_mech.jl")
    include("fig_flight.jl")
    
    # Supp-figs
    include("supp_fig_differential.jl")
    include("supp_fig_attractor_similarity.jl")
    include("supp_fig_wc_translate.jl")
    include("supp_fig_wc_transform.jl")
    include("supp_fig_wc_bifurcate.jl")
    include("supp_fig_translate_multi.jl")
    include("supp_fig_transform_multi.jl")
catch e
    println("An error occurred: $e")
    rethrow(e)
end

println("All scripts executed.")