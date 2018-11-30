#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <kdl/kdl.hpp>
#include <kdl/chain.hpp>
#include <kdl/tree.hpp>
#include <kdl/segment.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainiksolver.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/frames.hpp>
#include <kdl/frames_io.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/JointState.h>
#include "trajectory_msgs/JointTrajectory.h"
#include "/home/ctyou/ibvs_ros/devel/include/ibvs_core/ibvs_core.h"

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace KDL;

#define N 10   //提取的特征点数量

double joint_position[6] = {1, 1, 1, 1, 1, 1};
MatrixXd x_0(2 * N * 6, 1);                    //初始化的图像雅克比矩阵转化成的列向量
MatrixXd a = MatrixXd::Zero(2 * N, 1);
vector<MatrixXd> e_imgvel_con(3, a);

//高斯随机数产生器
double GenerateGaussianNum(double mean, double sigma)
{
    static double v1, v2, s;
    static int phase = 0;
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
void joint_position_callback(sensor_msgs::JointState msg)
{
    for(int i = 0; i < 6; i++)
        joint_position[i] = msg.position[i];
}

//主函数
int main(int argc, char ** argv)
{

    ros::init(argc, argv, "kalman");

    ros::NodeHandle n_ros;

    ros::Subscriber sub_joint_state = n_ros.subscribe("joint_states", 1000, joint_position_callback);
    ros::Publisher  vel_to_pub = n_ros.advertise<trajectory_msgs::JointTrajectory>("ur_driver/joint_speed", 1000);

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

    unsigned int nj = my_chain.getNrOfJoints();

    //计算目标图像特征
    cout << "开始计算目标图像特征" << endl;
    VideoCapture cap;
    cap.open(1);

    if(!cap.isOpened())
    {
        cerr << "couldn't open capture" << endl;
        return -1;
    }

    cap.set(CV_CAP_PROP_FPS, 60);
    bool bret;
    bret = cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    bret = cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    Mat src_desired,src_now;
    Mat src_desired_l, src_now_l;
    vector<KeyPoint> keypoints_desired, keypoints_now;
    Mat descriptors_desired, descriptors_now;
    Ptr<ORB> orb = ORB::create( N, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    src_desired = imread("/home/ctyou/visual-servo/visual-servo/build/1.jpg", 1);

    if(src_desired.empty())
    {
        cout << "src_desired is empty" << endl;
        return -1;
    }

    src_desired_l = src_desired.colRange(1,640).clone();
    orb -> detect(src_desired_l, keypoints_desired);
    orb -> compute(src_desired_l, keypoints_desired, descriptors_desired);

    MatrixXd x_k_ = MatrixXd::Zero(2 * N * 6,1);   //预估后的状态矩阵
    MatrixXd x_k = MatrixXd::Zero(2 * N * 6,1);    //校正后的状态矩阵
    MatrixXd A = MatrixXd::Identity(2 * N * 6, 2 * N * 6);
    MatrixXd h_k = MatrixXd::Zero(2 * N, 2 * N * 6);
    MatrixXd k_k = MatrixXd::Zero(2 * N * 6, 2 * N);      //卡尔曼增益矩阵
    MatrixXd p_k_ = MatrixXd::Identity(2 * N * 6, 2 * N * 6); //预估误差协方差矩阵
    MatrixXd p_k = MatrixXd::Identity(2 * N * 6, 2 * N * 6);  //校正后的误差协方差矩阵
    MatrixXd y_k = MatrixXd::Zero(2 * N, 1);          //观测值
    MatrixXd Q = MatrixXd::Identity(2 * N * 6, 2 * N * 6);
    MatrixXd R = MatrixXd::Identity(2 * N, 2 * N);

    MatrixXd rand = MatrixXd::Zero(2 * N, 1);

    ifstream infile;
    infile.open("/home/ctyou/ibvs_ros/src/ibvs_core/x_0.txt", ios::in);
    if(!infile.is_open()) {
        cout << "can not open!" << endl;
        return -1;
    }
    int x_0_count = 0;

    while(x_0_count != 2 * N * 6)
    {
        infile >> x_0(x_0_count, 0);
        x_0_count++;
    }

    for(int i = 0; i < 2 * N; i ++)
    {
        MatrixXd a(1,1);
        a(0,0) = GenerateGaussianNum(0, sqrt(10));
        rand(i, 0) = a(0, 0);
    }

    vector<MatrixXd> X_k_, X_k, P_k_, P_k, K_k, Y_k, H_k;
    H_k.push_back(h_k);
    X_k.push_back(x_0);
    X_k_.push_back(x_0);
    K_k.push_back(k_k);
    P_k.push_back(p_k);
    P_k_.push_back(p_k_);
    Y_k.push_back(y_k);

    JntArray joint_vel = JntArray(nj);
    MatrixXd joint_vel_ma_0 = MatrixXd::Zero(6, 1);
    MatrixXd joint_vel_ma_1 = MatrixXd::Zero(6, 1);
    MatrixXd joint_vel_ma_2 = MatrixXd::Zero(6, 1);

    ros::Rate loop_rate(1000);

    int j = 1;

    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
        //预测值
        x_k_ = A * X_k[j - 1];
        X_k_.push_back(x_k_);
        //预测误差协方差矩阵
        p_k_ = A * P_k[j - 1] * A.transpose() + Q;
        P_k_.push_back(p_k_);
        //开始更新
        k_k = P_k_[j] * H_k[j - 1].transpose() * (H_k[j - 1] * P_k_[j] * H_k[j - 1].transpose() + R).inverse();
        K_k.push_back(k_k);
        //测量值
        y_k = H_k[j - 1] * X_k[j - 1];
        Y_k.push_back(y_k);
        //估计值
        x_k = X_k_[j] + K_k[j] * (Y_k[j] - H_k[j - 1] * X_k_[j]);
        X_k.push_back(x_k);
        //估计误差协方差矩阵
        p_k = (MatrixXd::Identity(2 * N * 6, 2 * N * 6) - K_k[j] * H_k[j - 1]) * P_k_[j];
        P_k.push_back(p_k);

        int m_n = 0;
        MatrixXd img_jacobi(2 * N, 6);

        //计算当前时刻的图像雅可比矩阵
        for(int m = 0; m < 2 * N; m++)
        {
            for(int n = 0; n < 6; n++) {
                img_jacobi(m, n) = x_k(m_n, 0);
                m_n++;
            }
        }

        JacobiSVD<MatrixXd> svd(img_jacobi, ComputeFullU | ComputeFullV);

        int jac_row = img_jacobi.rows();
        int jac_col = img_jacobi.cols();
        int min_rc;
        if(jac_row > jac_col)
            min_rc = jac_col;
        else min_rc = jac_row;

        double pinvtoler = 1.e-8;
        MatrixXd singularValues_inv = svd.singularValues();
        MatrixXd singularValues_inv_mat = MatrixXd::Zero(jac_col, jac_row);

        for(int i = 0; i < min_rc; i++)
        {
            if(singularValues_inv(i) > pinvtoler)
                singularValues_inv(i) = 1.0 / singularValues_inv(i);
            else singularValues_inv(i) = 0;
        }

        for(int i = 0; i < min_rc; i++)
            singularValues_inv_mat(i, i) = singularValues_inv(i);


        MatrixXd jac_inv = (svd.matrixV()) * (singularValues_inv_mat) * (svd.matrixU().transpose()); //计算出的广义逆

        //开始计算当前的图像特征
        cap >> src_now;
        if(src_now.empty())
        {
            cout << "src_now is empty" << endl;
            return -1;
        }

        src_now_l = src_now.colRange(1,640).clone();
        //cvtColor(src_now_l, src_now_gray, CV_BGR2GRAY, 1);
        orb->detect(src_now_l, keypoints_now);
        orb->compute(src_now_l, keypoints_now, descriptors_now);

        vector<DMatch> matches;
        BFMatcher matcher (NORM_HAMMING);
        matcher.match(descriptors_desired, descriptors_now, matches);

        MatrixXd img_velocity(2 * N, 1);  //这就是图像特征差
        int point_count = 0;
        for(int i = 0; i < matches.size(); i++)
        {
            img_velocity(point_count, 0) = (keypoints_now[matches[i].trainIdx].pt.x - keypoints_desired[matches[i].queryIdx].pt.x);
            img_velocity(point_count + 1, 0) = (keypoints_now[matches[i].trainIdx].pt.y - keypoints_desired[matches[i].queryIdx].pt.y);
            point_count = point_count + 2;
        }

        cout << "img_velocity = " << endl;
        //cout << img_velocity << endl;

        e_imgvel_con[0] = img_velocity;

        cout << "jac_inv = " << endl;
        //cout << jac_inv << endl;

        double p_k = 0.01, i_k = 0.01, d_k = 0.01;
        JntArray joint_vel_delta = JntArray(nj);

        if(j == 1)
        {
            e_imgvel_con[1] = e_imgvel_con[0];
            e_imgvel_con[2] = e_imgvel_con[0];

            joint_vel_ma_0 = jac_inv * e_imgvel_con[0];

            joint_vel(0) = p_k * joint_vel_ma_0(0, 0);
            joint_vel(1) = p_k * joint_vel_ma_0(1, 0);
            joint_vel(2) = p_k * joint_vel_ma_0(2, 0);
            joint_vel(3) = p_k * joint_vel_ma_0(3, 0);
            joint_vel(4) = p_k * joint_vel_ma_0(4, 0);
            joint_vel(5) = p_k * joint_vel_ma_0(5, 0);
        }

        else
        {
            joint_vel_ma_0 = jac_inv * e_imgvel_con[0];
            joint_vel_ma_1 = jac_inv * e_imgvel_con[1];
            joint_vel_ma_2 = jac_inv * e_imgvel_con[2];

            joint_vel_delta(0) = p_k * (joint_vel_ma_0(0, 0) - joint_vel_ma_1(0, 0)) + i_k * joint_vel_ma_0(0, 0) + d_k * (joint_vel_ma_0(0, 0) - 2 * joint_vel_ma_1(0, 0) + joint_vel_ma_2(0, 0));
            joint_vel_delta(1) = p_k * (joint_vel_ma_0(1, 0) - joint_vel_ma_1(1, 0)) + i_k * joint_vel_ma_0(1, 0) + d_k * (joint_vel_ma_0(1, 0) - 2 * joint_vel_ma_1(1, 0) + joint_vel_ma_2(1, 0));
            joint_vel_delta(2) = p_k * (joint_vel_ma_0(2, 0) - joint_vel_ma_1(2, 0)) + i_k * joint_vel_ma_0(2, 0) + d_k * (joint_vel_ma_0(2, 0) - 2 * joint_vel_ma_1(2, 0) + joint_vel_ma_2(2, 0));
            joint_vel_delta(3) = p_k * (joint_vel_ma_0(3, 0) - joint_vel_ma_1(3, 0)) + i_k * joint_vel_ma_0(3, 0) + d_k * (joint_vel_ma_0(3, 0) - 2 * joint_vel_ma_1(3, 0) + joint_vel_ma_2(3, 0));
            joint_vel_delta(4) = p_k * (joint_vel_ma_0(4, 0) - joint_vel_ma_1(4, 0)) + i_k * joint_vel_ma_0(4, 0) + d_k * (joint_vel_ma_0(4, 0) - 2 * joint_vel_ma_1(4, 0) + joint_vel_ma_2(4, 0));
            joint_vel_delta(5) = p_k * (joint_vel_ma_0(5, 0) - joint_vel_ma_1(5, 0)) + i_k * joint_vel_ma_0(5, 0) + d_k * (joint_vel_ma_0(5, 0) - 2 * joint_vel_ma_1(5, 0) + joint_vel_ma_2(5, 0));

            joint_vel(0) = joint_vel(0) + joint_vel_delta(0);
            joint_vel(1) = joint_vel(1) + joint_vel_delta(1);
            joint_vel(2) = joint_vel(2) + joint_vel_delta(2);
            joint_vel(3) = joint_vel(3) + joint_vel_delta(3);
            joint_vel(4) = joint_vel(4) + joint_vel_delta(4);
            joint_vel(5) = joint_vel(5) + joint_vel_delta(5);
        }

        //截止条件
        if(e_imgvel_con[0].determinant() < 10)
        {
            cout << "到达目标位置！"<< endl;
            break;
        }

        MatrixXd e_temp(2 * N, 1);
        e_temp = e_imgvel_con[1];
        e_imgvel_con[1] = e_imgvel_con[2];
        e_imgvel_con[2] = e_temp;

        trajectory_msgs::JointTrajectory velocity_msg;
        velocity_msg.joint_names.resize(6);
        velocity_msg.joint_names[0]="shoulder_pan_joint";
        velocity_msg.joint_names[1]="shoulder_lift_joint";
        velocity_msg.joint_names[2]="elbow_joint";
        velocity_msg.joint_names[3]="wrist_1_joint";
        velocity_msg.joint_names[4]="wrist_2_joint";
        velocity_msg.joint_names[5]="wrist_3_joint";
        velocity_msg.header.stamp = ros::Time::now();
        velocity_msg.points.resize(1);
        velocity_msg.points[0].velocities.resize(6);
        velocity_msg.points[0].velocities[0] = joint_vel(0);
        velocity_msg.points[0].velocities[1] = joint_vel(1);
        velocity_msg.points[0].velocities[2] = joint_vel(2);
        velocity_msg.points[0].velocities[3] = joint_vel(3);
        velocity_msg.points[0].velocities[4] = joint_vel(4);
        velocity_msg.points[0].velocities[5] = joint_vel(5);

        for(int i = 0; i < 6; i++)
        {
            cout << "发送的第" << i << "个速度为： " << joint_vel(i) << endl;
        }

        if(abs((joint_vel(0))) >= 0.5 || abs((joint_vel(1))) >= 0.5 || abs((joint_vel(2))) >= 0.5 || abs((joint_vel(3))) >= 0.5 || abs((joint_vel(4))) >= 0.5 || abs((joint_vel(5))) >= 0.5)
            break;

        vel_to_pub.publish(velocity_msg);

        int vel_count = 0;
        for(int i = 0; i < 2 * N; i++)
        {
            for(int j_count = 0; j_count < 6; j_count++)
            {
                h_k(i, vel_count) = joint_vel(j_count);
                vel_count++;
            }
        }

        H_k.push_back(h_k);

        j++;
    }

    return 0;
}
