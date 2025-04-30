# common.py
import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(Nx, Ny):
    """初始化网格和边界条件"""
    T = np.full((Nx, Ny), 20.0, dtype=np.float64)  # 默认20°C
    T[:, -1] = 100.0  # 上边界设为100°C
    return T

def gauss_seidel_step(T, dx, dy):
    """执行一次高斯-赛德尔迭代"""
    residual = 0.0
    Nx, Ny = T.shape
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            temp = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])
            residual = max(residual, abs(temp - T[i,j]))
            T[i,j] = temp
    return residual

def sor_step(T, dx, dy, omega):
    """执行一次SOR迭代"""
    residual = 0.0
    Nx, Ny = T.shape
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            temp = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])
            temp = (1-omega)*T[i,j] + omega*temp
            residual = max(residual, abs(temp - T[i,j]))
            T[i,j] = temp
    return residual

def plot_temperature(T, dx, dy):
    """绘制等温线图"""
    x = np.arange(0, 15+dx, dx)
    y = np.arange(0, 12+dy, dy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    plt.contourf(X, Y, T, levels=20, cmap='jet')
    plt.colorbar(label='Temperature (°C)')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.title('Steady-State Temperature Distribution')
    plt.show()