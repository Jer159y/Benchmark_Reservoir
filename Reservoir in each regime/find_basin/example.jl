using DifferentialEquations
using DynamicalSystems

# 예시: Duffing 시스템
function duffing_rule(u, p, t)
    x, y = u
    δ, γ, ω = p
    return SVector{2}(y, -δ*y + x - x^3 + γ*cos(ω*t))
end

p = (0.2, 0.3, 1.2)
u0 = [0.1, 0.1]

# 시간 스팬
tmax = 500.0

ds = ContinuousDynamicalSystem(duffing_rule, u0, p)

# largest Lyapunov exponent
λ = lyapunov(ds, tmax)
println("Lyapunov exponent = ", λ)
