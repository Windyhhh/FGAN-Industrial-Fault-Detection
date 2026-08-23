import warnings
warnings.filterwarnings('ignore')
import numpy as np


# -----------------------
# PCA统计量计算
# -----------------------
def compute_pca_statistics(data, cpv=0.95):
    """计算PCA的T2和SPE统计量，不做归一化"""
    n_samples, n_features = data.shape
    cov_matrix = np.cov(data.T) + np.eye(n_features) * 1e-8
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = np.argmax(cumsum >= cpv) + 1

    principal_components = eigenvectors[:, :n_components]
    residual_components = eigenvectors[:, n_components:]
    principal_eigenvalues = eigenvalues[:n_components] + 1e-8

    scores = data @ principal_components
    T2 = np.sum((scores ** 2) / principal_eigenvalues, axis=1)

    if residual_components.shape[1] > 0:
        residuals = data @ residual_components
        SPE = np.sum(residuals ** 2, axis=1)
    else:
        SPE = np.zeros(n_samples)

    return np.column_stack([T2, SPE])


# -----------------------
# DPCA T2/SPE特征
# -----------------------
def DPCA_T2_SPE_feature(train_data, CPV, lag_number):
    """计算DPCA的T2和SPE统计量，不做归一化"""
    row_number, clumn_number = train_data.shape
    augmented_data = np.zeros((row_number - lag_number, (lag_number + 1) * clumn_number))
    for i in range(lag_number + 1):
        augmented_data[:, i * clumn_number:(i + 1) * clumn_number] = train_data[i:row_number - lag_number + i, :]
    DPCA_feature_traindata = compute_pca_statistics(augmented_data, CPV)
    return DPCA_feature_traindata


# -----------------------
# MD123特征
# -----------------------
def MD_feature(train_data):
    """计算马氏距离特征，不做归一化"""
    row_number, clumn_number = train_data.shape
    cov = np.cov(train_data.T) + np.eye(clumn_number) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigval_diag_inv = np.linalg.inv(np.diag(eigvals))
    MD_feature_traindata = np.zeros((row_number, 1))
    for i in range(row_number):
        MD_feature_traindata[i, 0] = np.sqrt(
            train_data[i, :].dot(eigvecs).dot(eigval_diag_inv).dot(eigvecs.T).dot(train_data[i, :].T))
    return MD_feature_traindata


def MD123_feature(train_data):
    """计算多部分马氏距离特征，不做归一化"""
    MD1 = MD_feature(train_data)
    MD2 = MD_feature(train_data[:, :22])
    MD3 = MD_feature(train_data[:, 22:])
    return np.hstack([MD1, MD2, MD3])


# -----------------------
# 滑动窗口二次型特征
# -----------------------
def sliding_window_quadratic_features(data, lag_number, cpvs=[0.99, 0.95, 0.90]):
    """使用滑动窗口计算二次型特征，不进行归一化"""
    n_samples, n_features = data.shape
    valid_samples = n_samples - lag_number

    # 构建滑动窗口数据
    windows = []
    for i in range(lag_number, n_samples):
        window = data[i - lag_number:i + 1, :].flatten()
        windows.append(window)
    windows = np.array(windows)

    window_features = []

    # 为每个CPV值计算二次型特征
    for cpv in cpvs:
        # 计算窗口数据的协方差矩阵
        win_cov = np.cov(windows.T) + np.eye(windows.shape[1]) * 1e-8
        eigvals, eigvecs = np.linalg.eigh(win_cov)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 确定主成分数量
        cum_var = np.cumsum(eigvals) / np.sum(eigvals)
        n_components = np.argmax(cum_var >= cpv) + 1

        # 分离主成分和残差成分
        principal_eigvals = eigvals[:n_components]
        principal_eigvecs = eigvecs[:, :n_components]
        residual_eigvecs = eigvecs[:, n_components:]

        # 计算T2统计量
        scores = windows @ principal_eigvecs
        T2 = np.sum((scores ** 2) / principal_eigvals, axis=1).reshape(-1, 1)

        # 计算SPE统计量
        if residual_eigvecs.shape[1] > 0:
            residuals = windows @ residual_eigvecs
            SPE = np.sum(residuals ** 2, axis=1).reshape(-1, 1)
        else:
            SPE = np.zeros((valid_samples, 1))

        # 合并T2和SPE，不进行归一化
        quad_features = np.hstack([T2, SPE])
        window_features.append(quad_features)

    # 合并所有CPV的特征
    return np.hstack(window_features)


# -----------------------
# 主特征提取函数
# -----------------------
def Feature_input_layer(train_data, CPV=0.99, lag_number=4):
    """主特征提取函数，不进行归一化以完全保留物理信息"""
    # 直接使用原始数据，不进行任何归一化
    n_samples, n_features = train_data.shape

    print(f"原始数据维度: {train_data.shape}")

    # 1. 滑动窗口二次型特征
    window_quad_features = sliding_window_quadratic_features(train_data, lag_number)
    print(f"滑动窗口二次型特征维度: {window_quad_features.shape}")

    # 2. DPCA T2/SPE特征
    DPCA_feature = DPCA_T2_SPE_feature(train_data, CPV, lag_number)
    print(f"DPCA特征维度: {DPCA_feature.shape}")

    # 3. PCA统计量
    pca_stats = compute_pca_statistics(train_data, CPV)
    # 对齐窗口但不归一化
    pca_stats_s = pca_stats[lag_number:]
    print(f"PCA统计量维度: {pca_stats_s.shape}")

    # 4. MD123特征
    md_features = MD123_feature(train_data)[lag_number:]  # 对齐窗口
    print(f"MD特征维度: {md_features.shape}")

    # 合并最终特征
    final = np.hstack([ window_quad_features, DPCA_feature, pca_stats_s, md_features])
    print(f"合并前特征维度: {final.shape}")

    # 仅处理异常值
    final = np.nan_to_num(final, nan=0.0, posinf=1e6, neginf=-1e6)
    print(f"最终特征维度: {final.shape}")


    return final

