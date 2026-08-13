import numpy as np
from sklearn.model_selection import train_test_split

# 1. 데이터 준비 (300개의 서로 다른 파라미터 c 생성)
n_samples = 300
c_range = np.linspace(0, 40, n_samples) # c [0, 40] 범위 [cite: 366]
all_features = []
all_targets = []

# 레저버 모델 초기화 (이전 단계에서 정의한 클래스 사용)
rc_model = ReservoirModel(n_nodes=300) 

print("데이터 생성 및 특징 추출 중...")
for c in c_range:
    # 시계열 생성 (이전 단계의 generate_lorenz 함수 사용)
    u = generate_lorenz(c_val=c) 
    # 특징 추출 (300개 노드 + 1개 편향 = 301차원) [cite: 947, 957]
    feature = rc_model.extract_features(u)
    
    all_features.append(feature)
    all_targets.append(c)

# 행렬 형태로 변환
X = np.array(all_features).T # (301, 300) 
Z = np.array(all_targets).reshape(1, -1) # (1, 300) [cite: 245]

# 2. 데이터 분할 (훈련 200, 검증 50, 테스트 50) [cite: 367]
# 여기서는 단순화를 위해 훈련(200)과 테스트(100)로 분할 예시
X_train, X_test, Z_train, Z_test = train_test_split(
    X.T, Z.T, train_size=200, random_state=42
)
X_train, X_test = X_train.T, X_test.T
Z_train, Z_test = Z_train.T, Z_test.T

# 3. 릿지 회귀 학습 (Readout 결정) [cite: 238, 941]
def train_readout(X_mat, Z_vec, lambd=1e-6):
    n_features = X_mat.shape[0]
    # R = Z * X.T * (X * X.T + lambda * I)^-1 
    identity = np.eye(n_features)
    reg_term = lambd * identity
    
    inv_part = np.linalg.inv(X_mat @ X_mat.T + reg_term)
    R = Z_vec @ X_mat.T @ inv_part
    return R

# 학습 실행
lambda_param = 1e-6 # 하이퍼파라미터 [cite: 240]
R_matrix = train_readout(X_train, Z_train, lambd=lambda_param)

# 4. 테스트 데이터 예측 및 평가 (NRMSE) [cite: 222, 352]
Z_pred = R_matrix @ X_test

def calculate_nrmse(actual, predicted):
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    # 정규화: 실제 값의 범위로 나눔 [cite: 223, 225]
    return rmse / (np.max(actual) - np.min(actual))

nrmse_score = calculate_nrmse(Z_test, Z_pred)
print(f"테스트 세트 NRMSE: {nrmse_score:.5f}")

# 결과 시각화 (Actual vs Extracted) [cite: 474]
import matplotlib.pyplot as plt
plt.scatter(Z_test, Z_pred, alpha=0.6, edgecolors='k')
plt.plot([0, 40], [0, 40], 'r--') # 대각선
plt.xlabel("Actual c (Lorenz)")
plt.ylabel("Extracted c")
plt.title(f"Parameter Extraction Results (NRMSE: {nrmse_score:.5f})")
plt.show()