using NLsolve
using Printf

function lorenz_fixed_points!(F, u, p)
    σ, ρ, β = p
    x, y, z = u
    F[1] = σ * (y - x)
    F[2] = x * (ρ - z) - y
    F[3] = x * y - β * z
end

σ = 10.0
β = 8.0 / 3.0
ρ = 28.0 # 혼돈 어트랙터와 3개의 불안정한 고정점이 있는 경우
p = [σ, ρ, β]

initial_guesses = [
    [0.1, 0.1, 0.1],      # 원점(Trivial) 근처에서 시작
    [10.0, 10.0, 10.0],   # C+ (Non-trivial) 근처에서 시작
    [-10.0, -10.0, -10.0] # C- (Non-trivial) 근처에서 시작
]

println("파라미터 (σ=$σ, ρ=$ρ, β=$β) 에서 고정점 자동 계산:")
println("-"^50)

found_solutions = []

for guess in initial_guesses
    f_to_solve! = (F, u) -> lorenz_fixed_points!(F, u, p)
    
    result = nlsolve(f_to_solve!, guess)
    
    if converged(result)
        solution = result.zero
        
        is_new = true
        for sol in found_solutions
            if isapprox(sol, solution, atol=1e-6)
                is_new = false
                break
            end
        end
        
        if is_new
            push!(found_solutions, solution)
            @printf "초기 추정값 [%.1f, %.1f, %.1f] -> 찾은 해: [%.4f, %.4f, %.4f]\n" guess[1] guess[2] guess[3] solution[1] solution[2] solution[3]
        end
    end
end

println("-"^50)
println("최종적으로 찾은 고정점 개수: $(length(found_solutions))")