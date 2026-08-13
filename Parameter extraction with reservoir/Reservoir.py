import numpy as np

class ReservoirModel:
    def __init__(self, n_nodes=300, input_dim=1, spectral_radius=1.0, degree=5):
        self.n_nodes = n_nodes
        
        # 1. Input weights (Standard Gaussian) 
        self.W_in = np.random.randn(n_nodes, input_dim)
        
        # 2. Internal weights (W) with average degree 
        W = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            # 노드당 5개의 연결만 랜덤하게 생성
            indices = np.random.choice(n_nodes, degree, replace=False)
            W[i, indices] = np.random.randn(degree)
            
        # 스펙트럼 반지름 조정 (Spectral Radius = 1.0) 
        current_radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / current_radius)
        
        # 3. Bias term (Standard Gaussian) 
        self.phi = np.random.randn(n_nodes)

    def extract_features(self, u):
        """
        입력 시계열 u(t)로부터 평균 레저버 활성화 상태를 추출함.
        u: (time_steps,) 형태의 넘파이 배열
        """
        time_steps = len(u)
        x = np.zeros(self.n_nodes) # 초기 상태
        
        # 모든 시점에서의 활성화 상태를 저장할 행렬
        states = np.zeros((time_steps, self.n_nodes))
        
        # 레저버 업데이트 루프 
        for t in range(time_steps):
            # x(t) = tanh(W*x(t-1) + W_in*u(t) + phi)
            input_part = self.W_in @ np.array([u[t]])
            x = np.tanh(self.W @ x + input_part + self.phi)
            states[t, :] = x
            
        # 논문의 핵심: 시간 축에 대한 평균값(Mean Activation)을 특징으로 사용 [cite: 264, 910]
        # 이 특징 벡터가 파라미터 추출의 입력이 됨.
        mean_feature = np.mean(states, axis=0)
        
        # 상수 편향 특징(Bias element of unit value) 추가 [cite: 956]
        return np.concatenate(([1.0], mean_feature))

# --- 실행 예시 ---
# 모델 초기화
model = ReservoirModel(n_nodes=300)

# 이전 단계에서 생성한 lorenz_data(예시)로부터 특징 추출
# lorenz_data는 (5000,) 크기의 배열이라고 가정
extracted_feature = model.extract_features(lorenz_data)

print(f"추출된 특징 벡터 크기: {extracted_feature.shape}") 
# (301,) -> 노드 300개 + 상수 편향 1개