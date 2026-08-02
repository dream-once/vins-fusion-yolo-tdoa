#pragma once
#include <ceres/ceres.h>
#include <Eigen/Dense>

class TDOAFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    // 构造函数：传入基站 i 的坐标、基站 j 的坐标，以及两者到无人机的距离差测量值
    TDOAFactor(const Eigen::Vector3d& _p_i, const Eigen::Vector3d& _p_j, double _measure_diff);

    // Ceres 求解器每次迭代计算残差和雅可比的回调函数
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

private:
    Eigen::Vector3d p_i;
    Eigen::Vector3d p_j;
    double measure_diff;
};