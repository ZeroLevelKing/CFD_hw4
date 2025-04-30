# task1.py
from common import initialize_grid, gauss_seidel_step, plot_temperature

def solve_steady_state(Nx=31, Ny=25, max_iter=10000, tol=1e-5):
    # 初始化参数
    dx = 15 / (Nx-1)
    dy = 12 / (Ny-1)
    
    # 初始化和边界条件
    T = initialize_grid(Nx, Ny)
    
    # 迭代求解
    for iter in range(max_iter):
        residual = gauss_seidel_step(T, dx, dy)
        if residual < tol:
            print(f"Converged in {iter+1} iterations")
            break
    else:
        print("Warning: Not converged")
    
    # 绘制结果
    plot_temperature(T, dx, dy)
    return T

if __name__ == "__main__":
    solve_steady_state()