# fig_intro.jl
using Plots, Printf, LinearAlgebra, SparseArrays, MAT, Random, Colors
using Plots.PlotMeasures # cm/px 단위 사용

# 헬퍼 모듈 포함
include("Lorenz.jl")
include("ReservoirTanh.jl")
include("downsample_curvature.jl")
include("plot_spline.jl")

println("fig_intro.jl (Julia Version) running...")

# == 1. Prepare Space ==
Random.seed!(1234) # 재현성을 위해 시드 설정

# == 2. Parameters and dimensions ==
FS = 10 # Fontsize
fSize = (19, 8.0) # Figure Size in cm

# 색상 정의
CL = [RGB(100/255, 100/255, 150/255), 
      RGB(100/255, 100/255, 170/255), 
      RGB(100/255, 100/255, 190/255)]
CO = [RGB(150/255, 100/255, 100/255), 
      RGB(170/255, 100/255, 100/255), 
      RGB(190/255, 100/255, 100/255)]
CR = [RGB(220/255, 200/255, 100/255), 
      RGB(220/255, 180/255, 100/255), 
      RGB(220/255, 160/255, 100/255)]
CPr = RGB(110/255, 190/255, 240/255)
CB = RGB(100/255, 100/255, 255/255)
CW = RGB(255/255, 100/255, 100/255)

# == 3. Training parameters ==
delT = 0.001
t_waste = 20
t_train = 200
n_w = Int(t_waste / delT)
n_t = Int(t_train / delT)
n = n_w + n_t
ind_t = (1:n_t) .+ n_w
t_ind = ind_t

# == 4. Initialize reservoir and Lorenz ==
N = 450
M = 3
gam = 100
sig = 0.008
c = 0.004
p = 0.1
x0 = zeros(M)
c0 = zeros(length(c))

# .mat 파일 로드
println("Loading fig_intro_params.mat...")
vars = matread("Teaching recurrent neural networks to infer global temporal structure from local examples/original/data/fig_intro_params.mat") # .mat 파일이 이 스크립트와 같은 위치에 있어야 함
A = vars["A"]
B = vars["B"]
C = vars["C"]
r0 = vec(vars["r0"]) # 벡터로 변환
Lx0 = vec(vars["Lx0"])

# 객체 생성
R2 = ReservoirTanh(A, B, C, r0, x0, c0, delT, gam)
L0 = Lorenz(Lx0, delT, [10.0, 28.0, 8.0/3.0])

# == 5. Lorenz time series ==
println("Simulating Attractor...")
X0 = propagate!(L0, n)

# == 6. Plotting Functions ==

# a: Plot training data
function plot_a(X0, ind_t, CL)
    println("Plotting a...")
    nPlot = 1:20000
    X0p = X0[:, ind_t[nPlot], 1]
    
    X0ps1, _ = downsample_curvature([nPlot'; X0p[1, :]'], 0.5)
    X0ps2, _ = downsample_curvature([nPlot'; X0p[2, :]'], 0.5)
    X0ps3, _ = downsample_curvature([nPlot'; X0p[3, :]' .- 27], 0.5)

    p = plot(X0ps1[1,:], X0ps1[2,:]/75 .+ 3, color=CL[1], lw=0.4, label=nothing)
    plot!(p, X0ps2[1,:], X0ps2[2,:]/75 .+ 2.25, color=CL[2], lw=0.4, label=nothing)
    plot!(p, X0ps3[1,:], X0ps3[2,:]/75 .+ 1.5, color=CL[3], lw=0.4, label=nothing)
    
    # x축 (시간축) 스플라인
    plot_spline!(p, [min(nPlot) max(nPlot)*0.98; 1 1], head=false, linewidth=0.3)
    plot_spline!(p, [min(nPlot) max(nPlot); 1 1], head=true, headpos=1.0, linewidth=0.3)

    plot!(p, title="a) Lorenz time series", 
          axis=nothing, border=:none, 
          ylim=(0.5, 4), xlim=(min(nPlot), max(nPlot)),
          titlefontsize=FS, fontfamily="LaTeX")
          
    # MATLAB: text(labX,2.35,'$x_1$',NVTitle{:});
    # Julia: annotate! 사용
    annotate!(p, min(nPlot), 2.35, text("x₁", :left, FS))
    annotate!(p, min(nPlot), 1.65, text("x₂", :left, FS))
    annotate!(p, min(nPlot), 0.95, text("x₃", :left, FS))
    annotate!(p, (min(nPlot)+max(nPlot))/2, 0.7, text("t", :center, FS))
    
    return p
end

# b: Plot reservoir schematic
function plot_b()
    println("Plotting b (placeholder)...")
    # 원본 MATLAB 코드는 매우 복잡한 수동 좌표와 스플라인으로 다이어그램을 그립니다.
    # (lines 142-205)
    # 이는 Plots.jl로 번역하기 매우 어렵고 Makie.jl나 Luxor.jl이 적합합니다.
    # 여기서는 재구현 대신 플레이스홀더를 반환합니다.
    p = plot(title="b) drive reservoir", 
             axis=nothing, border=:none, 
             titlefontsize=FS, fontfamily="LaTeX")
    annotate!(p, 0.5, 0.5, text("(Schematic Placeholder)", :center, 8, :grey))
    return p
end

# -- Reservoir Training --
println("Simulating Reservoir...")
RT_all = train!(R2, X0, zeros(1, n, 4)) #
RT = RT_all[:, t_ind]

println("Training W...")
# MATLAB: W = lsqminnorm(RT', X0(:,t_ind,1)')';
# Julia: W = (RT' \ X0[:, t_ind, 1]')'  (표준 \ 사용)
# X0[:, t_ind, 1] -> 3xN_t, RT -> N x N_t
W = (RT' \ X0[:, t_ind, 1]')'
XT = W * RT
train_error = norm(XT - X0[:, t_ind, 1])
println("Training error: $train_error") #

# c: Plot trained outputs
function plot_c(XT, CO)
    println("Plotting c...")
    nPlot = 1:20000
    XTp = XT[:, nPlot]

    XResds1, _ = downsample_curvature([nPlot'; XTp[1, :]'], 0.5)
    XResds2, _ = downsample_curvature([nPlot'; XTp[2, :]'], 0.5)
    XResds3, _ = downsample_curvature([nPlot'; XTp[3, :]' .- 27], 0.5)

    p = plot(XResds1[1,:], XResds1[2,:]/75 .+ 3, color=CO[1], lw=0.4, label=nothing)
    plot!(p, XResds2[1,:], XResds2[2,:]/75 .+ 2.25, color=CO[2], lw=0.4, label=nothing)
    plot!(p, XResds3[1,:], XResds3[2,:]/75 .+ 1.5, color=CO[3], lw=0.4, label=nothing)

    plot_spline!(p, [min(nPlot) max(nPlot)*0.98; 1 1], head=false, linewidth=0.3)
    plot_spline!(p, [min(nPlot) max(nPlot); 1 1], head=true, headpos=1.0, linewidth=0.3)
    
    plot!(p, title="c) train W", 
          axis=nothing, border=:none, 
          ylim=(0.5, 4), xlim=(min(nPlot), max(nPlot)),
          titlefontsize=FS, fontfamily="LaTeX")

    annotate!(p, min(nPlot), 2.35, text("x̂₁", :left, FS))
    annotate!(p, min(nPlot), 1.65, text("x̂₂", :left, FS))
    annotate!(p, min(nPlot), 0.95, text("x̂₃", :left, FS))
    annotate!(p, (min(nPlot)+max(nPlot))/2, 0.7, text("t", :center, FS))
    
    return p
end

# d: Plot feedback reservoir schematic
function plot_d()
    println("Plotting d (placeholder)...")
    # (Plot b와 동일한 사유로 플레이스홀더)
    p = plot(title="d) close feedback loop", 
             axis=nothing, border=:none, 
             titlefontsize=FS, fontfamily="LaTeX")
    annotate!(p, 0.5, 0.5, text("(Schematic Placeholder)", :center, 8, :grey))
    return p
end

# -- Predict time series --
R2.r = RT[:, n_t] # Initialize reservoir state
RP = predict_x!(R2, zeros(1, 40000, 4), W)
XP = W * RP

# e: Plot reservoir prediction versus Lorenz
function plot_e(XT, XP, CL, CPr)
    println("Plotting e...")
    XTp = XT[:, 1:20000] # 훈련 데이터 (짧게 자름)
    XPp = XP # 예측 데이터

    XTps, _ = downsample_curvature(XTp .- [0; 0; 29], 0.2, (10, 20))
    XPps, _ = downsample_curvature(XPp .- [0; 0; 29], 0.2, (10, 20))

    p = plot(XTps[1,:], XTps[2,:], XTps[3,:], 
             color=CL[2], lw=0.2, label="x (true)", 
             camera=(10, 20))
    plot!(p, XPps[1,:], XPps[2,:], XPps[3,:], 
          color=CPr, lw=0.2, label="x' (predicted)")
          
    # 축 그리기
    axSh = 20; axL = 40;
    plot!(p, [0, axL].-axSh, [0, 0].-axSh, [0, 0].-axSh, color=:black, lw=0.7, label=nothing)
    plot!(p, [0, 0].-axSh, [0, axL].-axSh, [0, 0].-axSh, color=:black, lw=0.7, label=nothing)
    plot!(p, [0, 0].-axSh, [0, 0].-axSh, [0, axL].-axSh, color=:black, lw=0.7, label=nothing)

    plot!(p, title="e) predicted output", 
          axis=nothing, border=:none, 
          titlefontsize=FS, fontfamily="LaTeX", legend=:topleft, legendfontsize=6)
    return p
end

# f: Plot Lorenz time series (shifted)
function plot_f(X0, ind_t, CL)
    println("Plotting f...")
    nPlot = 1:20000
    X0p = X0[:, ind_t[nPlot], 1] .- [0; 0; 27]
    pSh = 10
    X0D = hcat(X0p, X0p .+ [pSh; 0; 0], X0p .+ [2*pSh; 0; 0], X0p .+ [3*pSh; 0; 0])

    nPlotD = 1:(4*max(nPlot))
    X01Ds, _ = downsample_curvature([nPlotD'; X0D[1, :]'], 0.5)
    X02Ds, _ = downsample_curvature([nPlotD'; X0D[2, :]'], 0.5)
    X03Ds, _ = downsample_curvature([nPlotD'; X0D[3, :]'], 0.5)

    pSc = 75
    p = plot(X01Ds[1,:], X01Ds[2,:]/pSc .+ 3, color=CL[1], lw=0.4, label=nothing)
    plot!(p, X02Ds[1,:], X02Ds[2,:]/pSc .+ 2, color=CL[2], lw=0.4, label=nothing)
    plot!(p, X03Ds[1,:], X03Ds[2,:]/pSc .+ 1, color=CL[3], lw=0.4, label=nothing)
    
    # c (control) signal
    CN = palette(:winter, 10)
    XCDS = [0, 0, 1, 1, 2, 2, 3, 3] .* 0.15 .- 0.1
    nPlotC = [0, 1, 1, 2, 2, 3, 3, 4] .* max(nPlot)
    for i = 1:3
        plot!(p, nPlotC[(0:1).+2*i], XCDS[(0:1).+2*i], color=CN[i+3], lw=0.4, label=nothing)
    end
    plot!(p, nPlotC[7:8], XCDS[7:8], color=CN[6], lw=0.4, label=nothing)
    
    plot_spline!(p, [min(nPlotD) max(nPlotD); -0.2 -0.2], head=true, headpos=1.0, linewidth=0.1)
    
    plot!(p, title="f) training input: shifted Lorenz", 
          axis=nothing, border=:none, 
          ylim=(-0.6, 4.15), xlim=(min(nPlotD), max(nPlotD)),
          titlefontsize=FS, fontfamily="LaTeX")
          
    annotate!(p, min(nPlotD), 2.62, text("x₁", :left, FS))
    annotate!(p, min(nPlotD), 1.9, text("x₂", :left, FS))
    annotate!(p, min(nPlotD), 1.18, text("x₃", :left, FS))
    annotate!(p, min(nPlotD), 0.5, text("c", :left, FS))
    annotate!(p, (min(nPlotD)+max(nPlotD))/2, -0.4, text("t", :center, FS))
    
    return p
end

# g: Plot reservoir schematic (control)
function plot_g()
    println("Plotting g (placeholder)...")
    # (Plot b와 동일한 사유로 플레이스홀더)
    p = plot(title="g) drive reservoir (control)", 
             axis=nothing, border=:none, 
             titlefontsize=FS, fontfamily="LaTeX")
    annotate!(p, 0.5, 0.5, text("(Schematic Placeholder)", :center, 8, :grey))
    return p
end

# h: Plot trained outputs (shifted)
function plot_h(XT, CO)
    println("Plotting h...")
    nPlot = 1:20000
    pSh = 10
    XTP = XT[:, nPlot] .- [0; 0; 27]
    XTD = hcat(XTP, XTP .+ [pSh; 0; 0], XTP .+ [2*pSh; 0; 0], XTP .+ [3*pSh; 0; 0])

    nPlotD = 1:(4*max(nPlot))
    XT1Ds, _ = downsample_curvature([nPlotD'; XTD[1, :]'], 0.3)
    XT2Ds, _ = downsample_curvature([nPlotD'; XTD[2, :]'], 0.3)
    XT3Ds, _ = downsample_curvature([nPlotD'; XTD[3, :]'], 0.3)

    pSc = 75
    p = plot(XT1Ds[1,:], XT1Ds[2,:]/pSc .+ 3, color=CO[1], lw=0.4, label=nothing)
    plot!(p, XT2Ds[1,:], XT2Ds[2,:]/pSc .+ 2, color=CO[2], lw=0.4, label=nothing)
    plot!(p, XT3Ds[1,:], XT3Ds[2,:]/pSc .+ 1, color=CO[3], lw=0.4, label=nothing)

    plot_spline!(p, [min(nPlotD) max(nPlotD); -0.2 -0.2], head=true, headpos=1.0, linewidth=0.1)

    plot!(p, title="h) training output: shifted Lorenz", 
          axis=nothing, border=:none, 
          ylim=(-0.6, 4.15), xlim=(min(nPlotD), max(nPlotD)),
          titlefontsize=FS, fontfamily="LaTeX")

    annotate!(p, min(nPlotD), 2.62, text("x̂₁", :left, FS))
    annotate!(p, min(nPlotD), 1.9, text("x̂₂", :left, FS))
    annotate!(p, min(nPlotD), 1.18, text("x̂₃", :left, FS))
    annotate!(p, (min(nPlotD)+max(nPlotD))/2, -0.4, text("t", :center, FS))
    
    return p
end

# == 7. Generate and Combine Plots ==
println("Generating all plots...")

# 각 플롯 생성
pa = plot_a(X0, ind_t, CL)
pb = plot_b()
pc = plot_c(XT, CO)
pd = plot_d()
pe = plot_e(XT, XP, CL, CPr)
pf = plot_f(X0, ind_t, CL)
pg = plot_g()
ph = plot_h(XT, CO)

# 원본 MATLAB의 `subp` 레이아웃 재현
# (lines 28-35)
# 상단 5개 (a, b, c, d, e)
# 하단 3개 (f, g, h) - 너비 비율 (7.5, 4.0, 7.5)
# Plots.jl layout 매크로로 근사치 구현
layout = @layout [
    a b c d e
    f{0.416w} g{0.222w} h{0.416w}
]

# fSize를 cm 단위로 설정
fig_width_cm, fig_height_cm = fSize
# 1 cm = 37.8 pixels (근사치)
fig_size_px = (fig_width_cm * 37.8, fig_height_cm * 37.8)

# 모든 플롯 결합
final_plot = plot(pa, pb, pc, pd, pe, pf, pg, ph, 
                  layout=layout, 
                  size=fig_size_px, 
                  fontfamily="LaTeX")

# == 8. Save ==
fName = "fig_intro.pdf"
println("Saving figure to $fName ...")
savefig(final_plot, fName)
println("Done.")