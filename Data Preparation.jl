# 1. 시스템 시계열 생성 (Rössler 시스템 예시)
# X1, X2, X3는 상태 변수 (예: x, y, z)
# beta_1, beta_2, beta_3는 훈련에 사용할 파라미터 값 (예: bc = 0.6, 0.8, 1.0) [6]
data_set_1 = generate_timeseries(beta_1)
data_set_2 = generate_timeseries(beta_2)
data_set_3 = generate_timeseries(beta_3)

# 2. 입력 데이터 (Input Matrix) 구성 (시간적으로 연결)
U_train = hcat(data_set_1.states, data_set_2.states, data_set_3.states) # X(t)가 입력

# 3. 교사 데이터 (Teacher Matrix) 구성 (해당 파라미터 값으로 레이블링)
T_train_1 = fill(beta_1, size(data_set_1.states, 2))
T_train_2 = fill(beta_2, size(data_set_2.states, 2))
T_train_3 = fill(beta_3, size(data_set_3.states, 2))
Y_target = hcat(T_train_1, T_train_2, T_train_3) # beta가 출력