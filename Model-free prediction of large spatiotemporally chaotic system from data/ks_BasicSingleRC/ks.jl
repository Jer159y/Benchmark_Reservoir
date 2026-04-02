using Plots
using Random
using Statistics

# 필요한 파일들 포함 (같은 폴더에 있다고 가정)
include("generate_reservoir.jl")
include("kursiv_solve.jl")
include("reservoir_layer.jl")
include("train.jl")
include("train_reservoir.jl")
include("predict.jl")

# 메인 실행 블록
function main()
    # 파라미터 설정 (Dict 또는 NamedTuple 사용)
    # tau를 더 줄여서 수치 안정성 개선 (Kuramoto-Sivashinsky는 매우 강성(stiff)한 방정식)
    ModelParams = Dict(
        :tau => 0.001,          # 더 줄임 (100배 작은 시간 단계)
        :nstep => 100000,       # 총 시간 = 0.001 * 100000 = 100
        :N => 64,
        :d => 22
    )
    
    Random.seed!(1234) # Reproducibility를 위해 시드 설정 (MATLAB: rng('shuffle'))
    
    init_cond = 0.6 .* (-1 .+ 2 .* rand(Int(ModelParams[:N])))
    
    # 데이터 생성
    # kursiv_solve는 (Time x Space)를 반환하므로 transpose 처리 주의
    data_gen = kursiv_solve(init_cond, ModelParams)
    data = collect(data_gen') # (Space x Time) 형태로 변환
    
    # 데이터 검증
    if !all(isfinite, data)
        non_finite_count = count(!isfinite, data)
        inf_count = count(isinf, data)
        nan_count = count(isnan, data)
        println("ERROR: kursiv_solve generated non-finite data:")
        println("  Non-finite: $non_finite_count / $(length(data))")
        println("  Inf: $inf_count, NaN: $nan_count")
        println("  Data range: min=$(minimum(filter(isfinite, data))) max=$(maximum(filter(isfinite, data)))")
        error("Invalid data from kursiv_solve")
    end
    
    measured_vars = 1:Int(ModelParams[:N])
    measurements = data[measured_vars, :]
    
    # Reservoir 파라미터 설정
    num_inputs = size(measurements, 1)
    approx_res_size = 3000
    
    resparams = Dict(
        :radius => 0.6,
        :degree => 3,
        :N => floor(Int, approx_res_size/num_inputs) * num_inputs,
        :sigma => 0.5,
        :train_length => 70000,
        :num_inputs => num_inputs,
        :predict_length => 2000,
        :beta => 0.0001
    )
    
    println("Training Reservoir...")
    # 학습
    train_data = measurements[:, 1:Int(resparams[:train_length])]
    x_state, w_out, A, win = train_reservoir(resparams, train_data)
    
    println("Predicting...")
    # 예측 (Closed loop)
    # train의 마지막 상태(x_state)에서 시작하지만, 
    # MATLAB 코드는 predict 내부에서 x를 인자로 받아 바로 사용하지 않고
    # predict 함수 호출 시그니처: predict(A, win, resparams, x, H)
    # 여기서 x는 마지막 state.
    
    # 주의: MATLAB 코드 predict.m 에서는 루프 안에서 
    # x_aug를 만들고 out을 계산한 뒤, x를 업데이트함.
    # 즉, 입력받은 x(train 마지막 상태)를 기반으로 첫 예측을 수행.
    
    prediction, _ = predict(A, win, resparams, x_state, w_out)
    
    # --- Plotting ---
    println("Plotting results...")
    lambda_max = 0.05
    t_vals = (1:Int(resparams[:predict_length])) .* ModelParams[:tau] .* lambda_max
    s_vals = (1:Int(ModelParams[:N])) .* (60/128) # MATLAB 코드의 스케일링
    
    train_len = Int(resparams[:train_length])
    pred_len = Int(resparams[:predict_length])
    
    actual_data = data[:, train_len+1 : train_len + pred_len]
    error_data = actual_data .- prediction
    
    # 결과 통계 출력
    mse = mean((actual_data .- prediction).^2)
    mae = mean(abs.(actual_data .- prediction))
    println("\n=== 예측 성능 ===")
    println("MSE: $mse")
    println("MAE: $mae")
    println("Actual data range: [$(minimum(actual_data)), $(maximum(actual_data))]")
    println("Prediction range: [$(minimum(prediction)), $(maximum(prediction))]")
    println("Error range: [$(minimum(error_data)), $(maximum(error_data))]")
    
    # Heatmap 그리기는 주석 처리 (Plots 패키지 문제)
    # try
    #     p1 = heatmap(t_vals, s_vals, actual_data, 
    #         title="Actual", xlabel="Λ_max * t", c=:jet, clims=(-3,3))
    #     
    #     p2 = heatmap(t_vals, s_vals, prediction, 
    #         title="Prediction", xlabel="Λ_max * t", c=:jet, clims=(-3,3))
    #         
    #     p3 = heatmap(t_vals, s_vals, error_data, 
    #         title="Error", xlabel="Λ_max * t", c=:jet, clims=(-3,3))
    #     
    #     plot(p1, p2, p3, layout=(3,1), size=(600, 800))
    # catch e
    #     println("Plotting failed: $e")
    # end
end

# 실행
main()