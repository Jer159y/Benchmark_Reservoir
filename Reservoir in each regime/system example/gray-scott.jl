function gray_scott_1d!(du, u, p, t)
    N, Du, Dv, F, k, L = p
    # L은 1D Laplacian matrix for periodic boundary conditions
    
    U = @view u[1:N]
    V = @view u[N+1:2N]
    
    # 확산 항 계산
    diff_U = L * U
    diff_V = L * V
    
    # 반응 항 및 전체 미분 계산
    @. du[1:N]   = Du * diff_U - U * V^2 + F * (1 - U)
    @. du[N+1:2N] = Dv * diff_V + U * V^2 - (F + k) * V
end

# 2. 1D 라플라시안 행렬 생성 (주기적 경계 조건)
function create_1d_laplacian(N)
    L = Tridiagonal(fill(1.0, N-1), fill(-2.0, N), fill(1.0, N-1))
    L = Array(L)
    L[1, N] = 1.0 # Periodic boundary for the first row
    L[N, 1] = 1.0 # Periodic boundary for the last row
    return L
end

# 3. 시뮬레이션 실행 및 데이터 추출 함수
function get_gs_1d_data(N, Du, Dv, F, k, tspan, initial_perturbation=:random)
    L = create_1d_laplacian(N)
    
    u0 = zeros(2N)
    if initial_perturbation == :random
        # 초기 농도를 약간의 노이즈로 설정
        u0[1:N] .= 1.0 .+ 0.01 * randn(N) # U는 1에 가깝게
        u0[N+1:2N] .= 0.01 * randn(N)    # V는 0에 가깝게
    elseif initial_perturbation == :center_V # 중앙에 V를 집중시키는 초기 조건
        u0[1:N] .= 1.0
        u0[N+1:2N] .= 0.0
        center_idx = N ÷ 2
        u0[N+1+max(1, center_idx-5):N+1+min(N, center_idx+5)] .= 0.5
    end

    p = (N, Du, Dv, F, k, L)
    prob = ODEProblem(gray_scott_1d!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0, reltol=1e-6)
    
    # U 또는 V 중 하나의 농도 변화를 시각화 (여기서는 V)
    return reduce(hcat, [sol.u[t][N+1:2N] for t in 1:length(sol.t)])', sol.t
end

# ---------------------------------------------------------
# 4. 파라미터 셋업 (다양한 1D 패턴)
# ---------------------------------------------------------
N_gs_1d = 300
tspan_gs = (0.0, 3000.0) # 충분히 오래 시뮬레이션해야 패턴이 안정화됩니다.

# --- Case 1: Simple Stripes (반복적인 줄무늬) ---
# F, k 값에 따라 안정적인 줄무늬 패턴
Du1, Dv1, F1, k1 = 0.16, 0.08, 0.035, 0.065
data_gs1, t_gs1 = get_gs_1d_data(N_gs_1d, Du1, Dv1, F1, k1, tspan_gs, :center_V)

# --- Case 2: Soliton-like Patterns (고립파 같은 이동 패턴) ---
# 특정 F, k 값에서 패턴이 생성되고 이동함
Du2, Dv2, F2, k2 = 0.16, 0.08, 0.060, 0.062
data_gs2, t_gs2 = get_gs_1d_data(N_gs_1d, Du2, Dv2, F2, k2, tspan_gs, :center_V)

# --- Case 3: Chaotic Spot Generation (무질서한 점 생성 및 소멸) ---
# 1D에서도 스팟이 생기고 죽는 패턴을 볼 수 있음
Du3, Dv3, F3, k3 = 0.16, 0.08, 0.055, 0.062
data_gs3, t_gs3 = get_gs_1d_data(N_gs_1d, Du3, Dv3, F3, k3, tspan_gs, :center_V)

# ---------------------------------------------------------
# 5. 시각화 (1D Gray-Scott Subplots - CairoMakie)
# ---------------------------------------------------------
fig_gs = Figure(size=(1500, 500))
Label(fig_gs[0, :], "1D Gray-Scott Dynamics", fontsize=20)

ax1 = Axis(fig_gs[1, 1], title="1. GS 1D Stripes", xlabel="Time", ylabel="Space")
heatmap!(ax1, t_gs1, 1:N_gs_1d, data_gs1', colormap=:viridis)

ax2 = Axis(fig_gs[1, 2], title="2. GS 1D Solitons", xlabel="Time", ylabel="Space")
heatmap!(ax2, t_gs2, 1:N_gs_1d, data_gs2', colormap=:magma)

ax3 = Axis(fig_gs[1, 3], title="3. GS 1D Chaotic Spots", xlabel="Time", ylabel="Space")
heatmap!(ax3, t_gs3, 1:N_gs_1d, data_gs3', colormap=:thermal)

save("gray_scott_1d_dynamics.png", fig_gs)


# 2D Laplacian Matrix (with periodic boundary conditions)
function create_2d_laplacian(M)
    L_1d = Tridiagonal(fill(1.0, M-1), fill(-2.0, M), fill(1.0, M-1))
    L_1d = Array(L_1d)
    L_1d[1, M] = 1.0
    L_1d[M, 1] = 1.0
    
    L_2d = kron(sparse(I, M, M), L_1d) + kron(L_1d, sparse(I, M, M))
    return L_2d # Sparse matrix for efficiency
end

# Gray-Scott 2D Dynamics
function gray_scott_2d!(du, u, p, t)
    M_size, Du, Dv, F, k, L_2d = p
    N = M_size * M_size
    
    U = @view u[1:N]
    V = @view u[N+1:2N]
    
    # Diffusion
    diff_U = L_2d * U
    diff_V = L_2d * V
    
    # Reaction
    @. du[1:N]   = Du * diff_U - U * V^2 + F * (1 - U)
    @. du[N+1:2N] = Dv * diff_V + U * V^2 - (F + k) * V
end

# Simulation function for 2D GS
function get_gs_2d_animation(M, Du, Dv, F, k, tspan)
    L_2d = create_2d_laplacian(M)
    N = M*M
    
    u0 = zeros(2N)
    u0[1:N] .= 1.0 # U starts as 1 everywhere
    
    # Perturbation in the center for V
    mid = M ÷ 2
    for i in (mid-5):(mid+5), j in (mid-5):(mid+5)
        idx = (i-1)*M + j
        u0[N+idx] = 0.5 # Small V perturbation in the center
    end
    u0[N+1:2N] .+= 0.01 * randn(N) # Add some random noise to V

    p = (M, Du, Dv, F, k, L_2d)
    prob = ODEProblem(gray_scott_2d!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=10.0, reltol=1e-6) # Slower saveat for animation

    # Animation generation (plotting V concentration)
    anim = @animate for (i, t) in enumerate(sol.t)
        V_state = reshape(sol.u[i][N+1:2N], M, M)
        heatmap(V_state, zlim=(0.0, 1.0), title="GS 2D Pattern (t=$(Int(t)))", 
                c=:grays, aspect_ratio=:equal, axis=false)
    end
    gif(anim, "gs_2d_pattern.gif", fps=10)
end

# ---------------------------------------------------------
# 7. 2D Gray-Scott 시뮬레이션 (애니메이션)
# ---------------------------------------------------------
M_gs_2d = 100 # 100x100 격자
tspan_2d = (0.0, 5000.0)

# 유명한 "Worms" 패턴을 만드는 파라미터
Du_2d, Dv_2d, F_2d, k_2d = 0.16, 0.08, 0.060, 0.062 

# 또는 "Spots" 패턴: Du_2d, Dv_2d, F_2d, k_2d = 0.16, 0.08, 0.035, 0.065
# 또는 "Moving Spots": Du_2d, Dv_2d, F_2d, k_2d = 0.16, 0.08, 0.0545, 0.061

#get_gs_2d_animation(M_gs_2d, Du_2d, Dv_2d, F_2d, k_2d, tspan_2d)
# 이 함수를 주석 해제하고 실행하면 .gif 파일이 생성됩니다.