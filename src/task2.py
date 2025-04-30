# task2.py
import numpy as np
import matplotlib.pyplot as plt
from common import initialize_grid, sor_step

def compare_relaxation(omegas, Nx=31, Ny=25, max_iter=1000, tol=1e-5):
    # 初始化参数
    dx = 15 / (Nx-1)
    dy = 12 / (Ny-1)
    
    # 存储收敛历史
    convergence = {}
    
    for omega in omegas:
        T = initialize_grid(Nx, Ny)
        residuals = []
        
        for iter in range(max_iter):
            residual = sor_step(T, dx, dy, omega)
            residuals.append(residual)
            if residual < tol:
                break
        
        convergence[omega] = {
            'iterations': iter+1,
            'residuals': residuals
        }
        print(f"Omega={omega:.2f}: Converged in {iter+1} iterations")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10,6))
    for omega, data in convergence.items():
        residuals = data['residuals']
        plt.semilogy(residuals, label=f'ω={omega:.2f}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Residual (log scale)')
    plt.title('Convergence History with Different Relaxation Factors')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    omegas = [0.5, 1.0, 1.5, 1.7, 1.9]
    compare_relaxation(omegas)