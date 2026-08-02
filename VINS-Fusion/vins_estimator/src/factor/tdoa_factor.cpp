#include "tdoa_factor.h"

TDOAFactor::TDOAFactor(const Eigen::Vector3d& _p_i, const Eigen::Vector3d& _p_j, double _measure_diff)
    : p_i(_p_i), p_j(_p_j), measure_diff(_measure_diff)
{
}

bool TDOAFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // parameters[0] 包含了无人机的全局位姿 Pose (Px, Py, Pz, Qx, Qy, Qz, Qw)
    // 提取出当前的 3D 平移坐标 p_w
    Eigen::Vector3d p_w(parameters[0][0], parameters[0][1], parameters[0][2]);

    // 计算当前估计位置到两个基站的距离
    double d_i = (p_w - p_i).norm();
    double d_j = (p_w - p_j).norm();

    // 1. 计算残差 (Residual)
    residuals[0] = (d_i - d_j) - measure_diff;

    // 2. 计算雅可比矩阵 (Jacobian)
    if (jacobians)
    {
        if (jacobians[0])
        {
            // 将内存映射为 1 行 7 列的 Eigen 矩阵，方便操作
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
            
            // 初始化全为 0（这顺便把后面 4 维的旋转偏导数也设成了 0，因为测距与姿态无关）
            jacobian_pose.setZero(); 

            // 防止无人机刚好和基站坐标重合导致除以 0 的奇异性问题
            if (d_i > 1e-5 && d_j > 1e-5) 
            {
                // 对平移 p_w 的偏导数 (1x3 向量)
                Eigen::Vector3d dp_w = (p_w - p_i) / d_i - (p_w - p_j) / d_j;
                
                // 将算出的 1x3 导数填入雅可比矩阵的前 3 列
                jacobian_pose.block<1, 3>(0, 0) = dp_w.transpose();
            }
        }
    }

    return true;
}