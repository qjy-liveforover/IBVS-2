#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include "std_msgs/String.h"
#include "/home/ctyou/ibvs_ros/devel/include/ibvs_core/ibvs_core.h"

using namespace std;
using namespace Eigen;

#define N 50   //提取的特征点数量

double joint_position[6] = {1, 1, 1, 1, 1, 1};
MatrixXd x_0(2 * N * 6, 1);     //初始化的图像雅克比矩阵转化成的列向量

//MatrixXd P_0_ = MatrixXd::Zero(N * 6, N * 6);

//订阅函数：X_0和P_0初始化数据
//void init_par_callback(ibvs_core::ibvs_core msg)
//{
//for(int i = 0; i < N; i ++)
//x_0(i, 0) = msg.x_init[i];
//}

//高斯随机数产生器
double GenerateGaussianNum(double mean, double sigma)
{

   static double v1, v2, s;
   static int phase = 0;
   double RAND_MAX = 1.0;
   double x;

    if(phase == 0)
    {

    do{
    double u1 = (double)rand()/RAND_MAX;
    double u2 = (double)rand()/RAND_MAX;

    v1 = 2 * u1 - 1;
    v2 = 2 * u2 - 1;

    s= v1 * v1 + v2 * v2; }while(s >= 1 || s == 0);

    x = v1 * sqrt(-2 * log(s) / s);
    }

    else x = v2 * sqrt(-2 * log(s) / s);

    phase = 1 - phase;

    return (x * sigma + mean);
}

//机器人位置函数
void joint_position_callback(sensor_msg::JointState msg)
{
    for(int i = 0; i < 5; i++)
        joint_position[i] = msg.position[i];
}


//主函数
int main(int argc, char ** argv)
{

    Tree my_tree;
    if(!kdl_parser::treeFromFile("/home/ctyou/ibvs_ros/src/ur/universal_robot-kinetic-devel/ur_description/urdf/ur3_robot.urdf", my_tree))
    {
        ROS_ERROR("Failed to construct kdl tree");
    }

    bool exit_value;
    Chain my_chain;
    exit_value = my_tree.getChain("base_link", "ee_link", my_chain);
    if(!exit_value)
        ROS_ERROR("Failed to convert to chain");

    //计算目标图像特征
    cout << "开始计算目标图像特征，请准备就绪，准备就绪后，请按任意键！" << endl;


    MatrixXd x_k_ = MatrixXd::Zero(2 * N * 6,1);   //预估后的状态矩阵
    MatrixXd x_k = MatrixXd::Zero(2 * N * 6,1);    //校正后的状态矩阵
    MatrixXd A(2 * N, 2 * N);
    A << MatrixXd::Identity(2 * N, 2 * N);
    MatrixXd H = MatrixXd::Zero(2 * N, 2 * N * 6);
    MatrixXd k_k = MatrixXd::Zero(2 * N * 6, 2 * N);      //卡尔曼增益矩阵
    MatrixXd p_k_ = MatrixXd::Zero(2 * N * 6, 2 * N * 6); //预估误差协方差矩阵
    MatrixXd p_k = MatrixXd::Zero(2 * N * 6, 2 * N * 6);  //校正后的误差协方差矩阵
    MatrixXd y_k = MatrixXd::Zero(2 * N, 1);          //观测值
    MatrixXd Q = MatrixXd::Identity(2 * N * 6, 2 * N * 6);
    MatrixXd R = MatrixXd::Identity(2 * N, 2 * N);

    MatrixXd rand = MatrixXd::Zero(2 * N, 1);

    for(int i = 0; i < 2 * N; i ++)
    {
    MatrixXd a(1,1);
    a(0,0) = GenerateGaussianNum(0, sqrt(10));
    rand(i, 0) = a(0, 0);
    }

    vector<MatrixXd> X_k_, X_k, P_k_, P_k, K_k, Y_k;
//X_k_.push_back(x_0);
    X_k.push_back(x_0);
//P_k_.push_back();
//P_k.push_back();

    ros::init(argc, argv, "kalman");
    ros::NodeHandle n;

    ros::Subscriber sub_joint_state = n.subscribe("kalman", 100, jpint_position_callback);

    ros::Rate loop_rate(100);

    int j = 1;

    while(ros::ok())
    {
//预测值
    x_k_ = A * X_k[j - 1];
    X_k_.push_back(x_k_);
//预测误差协方差矩阵
    p_k_ = A * P_k[j - 1] * A.transpose() + Q;
    P_k_.push_back(p_k_);
//开始更新
    k_k = P_k_[j] * H.transpose() * (H * P_k_[j] * H.transpose() + R).inverse();
    K_k.push_back(k_k);
//测量值
    y_k = H * X_k[j] + rand;     //此处的H也需要订阅,为啥？
    Y_k.push_back(y_k);
//估计值
    x_k = X_k_[j] + K_k[j] * (Y_k[j] - H * X_k_[j]);
    X_k.push_back(x_k);
//估计误差协方差矩阵
    p_k = (MatrixXd::Identity(N, N) - K_k[j] * H) * P_k_[j];
    P_k_.push_back(p_k);

    int m_n = 0;
    MatrixXd img_jacobi(2 * N, 6);

   //计算当前时刻的图像雅克比
    for(int m = 0; m < 2 * N; m++)
    {
       for(int n = 0; n < 6; n++) {
           img_jacobi(m, n) = x_k(m_n, 0);
           m_n++;
       }
    }

    ChainJntToJacSolver jacsolver = ChainToJacSolver(my_chain);
    JntArray joint_pos(6);
    for(int i = 0; i < 5; i++)
    {
        joint_pos(i) = joint_position[i];
    }

    //求当前时刻的机器人雅克比矩阵
    bool kinematics_status;
    MatrixXd ro_jacobi(6, 6);
    Jacobian ro_jac;
    kinematics_status = jacsolver.JntToJac(joint_pos, ro_jac, -1);

    for(int i = 0; i < 6; i++){
        for(int j = 0; j < 6; j++){
            ro_jacobi(i, j) = ro_jac(i, j);
        }
    }

    MatrixXd jac_all = img_jacobi * ro_jacobi;
    JacobiSVD<MatrixXd> svd(jac_all, ComputeFullu | computeFullv);
    int asize = svd.singularValues().size();
    MatrixXd values = MatrixXcd::Zero(asizze, asize);
    for(int i = 0; i < asize; i++)
    {
        values(i, i) = svd.singularValues()(i);
    }
    int jac_row = jac_all.rows();
    int jac_col = jac_all.cols();
    MatrixXd S_inv = MatrixXd::Zero(jac_col, jac_row);
    S_inv.block(0, 0, asize, asize) = values.inverse();

    MatrixXd jac_inv = (svd.matrixV()) * (S_inv) * (svd.matrixU().transpose()); //计算出的广义逆

    ros::spinOnce();
    loop_rate.sleep();
    j++;
}

return 0;
}
