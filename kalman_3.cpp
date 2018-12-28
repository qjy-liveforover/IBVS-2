//
// Created by ctyou on 18-12-19.
//
//检测四个特征点
//如何判断两帧之间的特征点的对应关系：
//当能检测到4个点时
//上一帧的某一点与当前帧的所有点做坐标差，差最小的为对应的特征点，同时最小差为图像空间速度
//如何让初始帧与目标帧的特征点对应起来:假设初始位置相机绕Z轴旋转不超过90度
//根据以上假设，第一个点一定x坐标最小的两个点中y坐标最小的
//当检测到多余4个点坐标时，计算几个点坐标的坐标向量，相近的取坐标平均值
//当检测到少于4个点坐标时，未检测到的点的坐标为其余点坐标的平均增量
//当检测到少于等于2个坐标点时，记录一个数+1,当这个数大于30时，终止实验。
//目标点排序为：
// 1  - - - - -  2
// -             -
// -             -
// -             -
// 3  - - - - -  4

#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <fstream>

#include <opencv2/opencv.hpp>
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

#include "/home/ctyou/ibvs_ros/src/ibvs_core/src/EllipseDetectorYaed.h"

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace KDL;

#define N 4   //提取的特征点数量

double joint_position[6] = {1, 1, 1, 1, 1, 1};
MatrixXd x_0(2 * N * 6, 1);                    //初始化的图像雅克比矩阵转化成的列向量
MatrixXd a = MatrixXd::Zero(2 * N, 1);
vector<MatrixXd> e_imgvel_con(3, a);

//关键点排序用数据结构
struct kp_stru{
    double cd;     //x坐标
    int num;       //序号
};

bool comp(const kp_stru &kp1, const kp_stru &kp2)
{
    return kp1.cd <= kp2.cd;
}

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

//对同一个椭圆检测出多个椭圆进行处理
vector<Point> process_kp(vector<Point> kp)
{
    double radio = 3000; //自定义的重复椭圆圆心之间的最大距离，根据实际情况改变
    int n = kp.size();

    vector<double> cd_2;
    vector<double> cd_diff;

    for(int i = 0; i < n; i++)
    {
        cd_2.push_back(kp[i].x * kp[i].x + kp[i].y * kp[i].y);
    }

    for(int i = 0; i < n - 1; i++) {
        cd_diff.push_back(cd_2[i + 1] - cd_2[i]);
        if(abs(cd_diff[i]) < radio) {
            cd_diff[i] = 0;
        }
    }

    vector<Point> kp_end;
    for(int i = 0; i < n - 1; i++)
    {
        if(cd_diff[i] == 0) {
            int x_temp, y_temp;
            int j = i;

            x_temp = kp[i].x + kp[i + 1].x;
            y_temp = kp[i].y + kp[i + 1].y;

            while(cd_diff[++i] == 0)
            {
                x_temp += kp[i + 1].x;
                y_temp += kp[i + 1].y;
            }
            x_temp = x_temp / (i - j + 1);
            y_temp = y_temp / (i - j + 1);
            kp_end.push_back(Point(x_temp, y_temp));
        }
        else
        {
            kp_end.push_back(kp[i]);
        }
    }
    if( n >= 2 && ((kp[n - 1].x * kp[n - 1].x + kp[n - 1].y * kp[n - 1].y) - (kp[n - 2].x * kp[n - 2].x + kp[n - 2].y * kp[n - 2].y) > radio));
    kp_end.push_back(kp[n - 1]);
    return kp_end;
}

//对检测到的特征点进行排序
vector<Point> sort_keypoints( vector<Point> kp )
{
    int n = kp.size();

    vector<Point> sort_kp(n, Point(0, 0));
    vector<kp_stru> vector_kp;

    for(int i = 0; i < n; i ++)
    {
        kp_stru one_kp;
        one_kp.cd = kp[i].x;
        one_kp.num = i;
        vector_kp.push_back(one_kp);
    }

    for(int i = 0; i < n; i++)
        cout << "原始的x坐标和序号： " << vector_kp[i].cd << "  " << vector_kp[i].num << endl;

    sort(vector_kp.begin(), vector_kp.end(), comp);

    for(int i = 0; i < n; i++)
        cout << "sort之后的x坐标和序号： " << vector_kp[i].cd << "  " << vector_kp[i].num << endl;

    sort_kp[0] = kp[vector_kp[0].num].y < kp[vector_kp[1].num].y ? kp[vector_kp[0].num] : kp[vector_kp[1].num];
    sort_kp[2] = kp[vector_kp[0].num].y > kp[vector_kp[1].num].y ? kp[vector_kp[0].num] : kp[vector_kp[1].num];
    sort_kp[1] = kp[vector_kp[2].num].y < kp[vector_kp[3].num].y ? kp[vector_kp[2].num] : kp[vector_kp[3].num];
    sort_kp[3] = kp[vector_kp[2].num].y > kp[vector_kp[3].num].y ? kp[vector_kp[2].num] : kp[vector_kp[3].num];

    for(int i = 0; i < n; i++)
        cout << sort_kp[i] << endl;

    return sort_kp;
}

//主函数
int main(int argc, char ** argv)
{
    ros::init(argc, argv, "kalman_3");

    ros::NodeHandle n_ros;

    ros::Subscriber sub_joint_state = n_ros.subscribe("joint_states", 100, joint_position_callback);
    ros::Publisher  vel_to_pub = n_ros.advertise<trajectory_msgs::JointTrajectory>("ur_driver/joint_speed", 100);

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

    Mat3b src_desired;
    Mat3b src_desired_l;

    src_desired = imread("/home/ctyou/ibvs_ros/src/img_data/1.jpg", 1);
    src_desired_l = src_desired.colRange(1,640).clone();

    Size sz = src_desired_l.size();

    if(src_desired.empty())
    {
        cout << "src_desired is empty" << endl;
        return -1;
    }

    // Parameters Settings (Sect. 4.2)
    int	    iThLength = 16;                   //开始设置椭圆检测参数
    float	fThObb = 3.0f;
    float	fThPos = 1.0f;
    float	fTaoCenters = 0.05f;
    int 	iNs = 16;
    float	fMaxCenterDistance = sqrt(float(sz.width*sz.width + sz.height*sz.height)) * fTaoCenters;

    float	fThScoreScore = 0.7f;

    // Other constant parameters settings.
    // Gaussian filter parameters, in pre-processing
    Size	szPreProcessingGaussKernelSize = Size(5, 5);
    double	dPreProcessingGaussSigma = 1.0;

    float	fDistanceToEllipseContour = 0.1f;	// (Sect. 3.3.1 - Validation)
    float	fMinReliability = 0.5;	                // Const parameters to discard bad ellipses

    CEllipseDetectorYaed* yaed = new CEllipseDetectorYaed();
    yaed->SetParameters(
            szPreProcessingGaussKernelSize,
            dPreProcessingGaussSigma,
            fThPos,
            fMaxCenterDistance,
            iThLength,
            fThObb,
            fDistanceToEllipseContour,
            fThScoreScore,
            fMinReliability,
            iNs
    );

    // Detect
    vector<Ellipse> ellsYaed_desired;
    Mat1b src_desired_l_gray;

    cvtColor(src_desired_l, src_desired_l_gray, CV_BGR2GRAY);
    yaed->Detect(src_desired_l_gray, ellsYaed_desired);

    Mat3b draw_desired = src_desired_l.clone();
    yaed->DrawDetectedEllipses(draw_desired, ellsYaed_desired);

    imshow("desired image", draw_desired);
    waitKey(5);

    vector<Point> desired_points;
    for(int i = 0; i < ellsYaed_desired.size(); i++)
    {
        desired_points.push_back(Point(ellsYaed_desired[i]._xc, ellsYaed_desired[i]._yc));
        //cout << ellsYaed_desired[i]._xc << " " << ellsYaed_desired[i]._yc << endl;
    }
    cout << "  " << endl;
    cout << "desired_points.size()" << desired_points.size() << endl;
    desired_points = process_kp(desired_points);
    cout << "desired_points.size()" << desired_points.size() << endl;

    for(int i = 0; i < desired_points.size(); i++)
    {
        cout << "desired_x = " << desired_points[i].x << "  desired_y = " << desired_points[i].y << endl;
    }

    if(desired_points.size() != 4)
    {
        cout << "目标图像不能检测到4个特征点！ 请重新选择！" << endl;
        return -1;
    }

    desired_points = sort_keypoints(desired_points);

    //卡尔曼估计相关变量
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

    //读取初始化的雅可比矩阵
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

    //卡尔曼估计的初始值
    vector<MatrixXd> X_k_, X_k, P_k_, P_k, K_k, Y_k, H_k;
    X_k.push_back(x_0);
    X_k_.push_back(x_0);
    K_k.push_back(k_k);
    P_k.push_back(p_k);
    P_k_.push_back(p_k_);
    Y_k.push_back(y_k);

    JntArray joint_vel = JntArray(nj);
    MatrixXd joint_vel_ma_0 = MatrixXd::Zero(6, 1);

    //打开摄像头，设置采集的图像的参数
    VideoCapture cap;
    cap.open(1);

    if (!cap.isOpened()) {
        cerr << "couldn't open capture" << endl;
        return -1;
    }

    cap.set(CV_CAP_PROP_FPS, 60);
    bool bret;
    bret = cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    bret = cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    //检测初始帧的特征点
    Mat src_now;
    cap >> src_now;
    if(src_now.empty())
    {
        cout << "src_now is empty!" << endl;
        return -1;
    }

    Mat3b src_now_l = src_now.colRange(1,640).clone();
    Mat1b src_now_l_gray;
    vector<Ellipse> ellsYaed_now;
    vector<Point> now_points;

    cvtColor(src_now_l, src_now_l_gray, CV_BGR2GRAY, 1);
    yaed->Detect(src_now_l_gray, ellsYaed_now);

    Mat3b draw_src = src_now_l.clone();
    yaed->DrawDetectedEllipses(draw_src, ellsYaed_now);

    imshow("src_now", draw_src);
    waitKey(1);
    cout << " " << endl;
    cout << "初始帧检测椭圆个数为： " << ellsYaed_now.size() << endl;

    for(int i = 0; i < ellsYaed_now.size(); i++)
    {
        now_points.push_back(Point(ellsYaed_now[i]._xc, ellsYaed_now[i]._yc));
    }

    now_points = process_kp(now_points);
    if(now_points.size() != 4)
    {
        cout << "初始帧不能检测到4个特征点，请重新选取！" << endl;
        return -1;
    }

    sort_keypoints(now_points);

    //记录初始时刻的角度值
    ros::spinOnce();
    MatrixXd now_joint_pos(6, 1);
    for(int i = 0; i < 6; i++)
        now_joint_pos(i, 0) = joint_position[i];

    //broyden估计雅可比矩阵及其参数
    double lambda = 0.1;
    MatrixXd pre_rsl = MatrixXd::Identity(6, 6);

    int m_n_count = 0;
    MatrixXd pre_jacobian(2 * N, 6);
    MatrixXd img_jacobi(2 * N, 6);
    for(int m = 0; m < 2 * N; m++)
    {
        for(int n = 0; n < 6; n++) {
            img_jacobi(m, n) = x_0(m_n_count, 0);
            m_n_count++;
        }
    }

    ros::Rate loop_rate(100);

    int loop_count = 1;

    while(ros::ok())
    {
        //首先发送关节速度
        //计算图像雅可比矩阵
//        int m_n = 0;
//        MatrixXd img_jacobi(2 * N, 6);
//        for(int m = 0; m < 2 * N; m++)
//        {
//            for(int n = 0; n < 6; n++) {
//                img_jacobi(m, n) = X_k[loop_count - 1](m_n, 0);
//                m_n++;
//            }
//        }

        //计算图像雅可比矩阵的逆
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

        //计算图像空间速度
        MatrixXd img_velocity = MatrixXd::Zero(2 * N, 1);  //这就是图像特征差
        int point_count = 0;
        for(int i = 0; i < N; i++)
        {
            img_velocity(point_count, 0) = (now_points[i].x - desired_points[i].x);
            img_velocity(point_count + 1, 0) = (now_points[i].y - desired_points[i].y);
            point_count = point_count + 2;
        }

        for(int i = 0; i < 2 * N; i++)
        {
            cout << "与期望坐标相差： " << img_velocity(i, 0) << endl;
        }

        //截止条件
        for(int i = 0; i < 2 * N; i++)
        {
            if(abs(img_velocity(i, 0)) <= 20) {
                cout << "i = " << i << endl;
            }
            else
                break;

            if(i == 2 * N - 1)
            {
                cout << "达到目标位置，机器人停止工作！" << endl;
                return 1;
            }
        }

        //计算机器人关节速度并发送
        double pid_p = 0.05;
        JntArray joint_vel_delta = JntArray(nj);

        joint_vel_ma_0 = - jac_inv * img_velocity;

        joint_vel(0) = pid_p * joint_vel_ma_0(0, 0);
        joint_vel(1) = pid_p * joint_vel_ma_0(1, 0);
        joint_vel(2) = pid_p * joint_vel_ma_0(2, 0);
        joint_vel(3) = pid_p * joint_vel_ma_0(3, 0);
        joint_vel(4) = pid_p * joint_vel_ma_0(4, 0);
        joint_vel(5) = pid_p * joint_vel_ma_0(5, 0);

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
            if(joint_vel(i) > 1.5)
            {
                cout << "关节超速！" << endl;
                //return -1;
            }
        }
        vel_to_pub.publish(velocity_msg);

        //记录上一时刻的特征点和角度值
        vector<Point> pre_kp;
        pre_kp = now_points;
        MatrixXd pre_joint_pos(6, 1);
        pre_joint_pos = now_joint_pos;

        //清除一些vector向量
        vector<Point>().swap(now_points);
        vector<Ellipse>().swap(ellsYaed_now);

        //记录当前运动后的角度值和特征点
        for(int i = 0; i < 6; i++)
            now_joint_pos(i, 0) = joint_position[i];

        Mat src_now;
        cap >> src_now;
        if(src_now.empty())
        {
            cout << "src_now is empty!" << endl;
            return -1;
        }

        src_now_l = src_now.colRange(1,640).clone();

        cvtColor(src_now_l, src_now_l_gray, CV_BGR2GRAY, 1);
        yaed->Detect(src_now_l_gray, ellsYaed_now);

        draw_src = src_now_l.clone();
        yaed->DrawDetectedEllipses(draw_src, ellsYaed_now);

        imshow("src_now", draw_src);
        waitKey(1);
        //cout << " " << endl;
        //cout << "初始帧检测椭圆个数为： " << ellsYaed_now.size() << endl;

        for(int i = 0; i < ellsYaed_now.size(); i++)
        {
            now_points.push_back(Point(ellsYaed_now[i]._xc, ellsYaed_now[i]._yc));
        }

        now_points = process_kp(now_points);
        vector<Point> temp_now_points(4, Point(0, 0));

        int dis_radio = 50; //当两帧检测的特征点差小于这个阈值时，则认为检测到的为同一个特征点,该值需要验证
        if(now_points.size() == 4)
        {
            cout << "检测出4个点并进行排序处理" << endl;
            now_points = sort_keypoints(now_points);
        }
        else
        {
            cout << "检测的少于4个点开始借取上一帧检测的特征点！" << endl;
            for(int i = 0; i < pre_kp.size(); i++)
                cout << "pre_kp[i] = " << pre_kp[i] << endl;

            for(int i = 0; i < now_points.size(); i++)
            {
                for(int j = 0; j < pre_kp.size(); j++)
                {
                    if((abs(now_points[i].x - pre_kp[j].x) <= dis_radio) && (abs(now_points[i].y - pre_kp[j].y) <= dis_radio)) {
                        temp_now_points[j] = now_points[i];
                        break;
                    }
                }
                cout << "i = " << i << endl;
            }

            for(int i = 0; i < 4; i++)
                cout << "temp_now_points[i] = " << temp_now_points[i] << endl;

            double x_temp_sum = 0, y_temp_sum = 0;
            int num_count_temp = 0;
            for(int i = 0; i < pre_kp.size(); i++)
            {
                if(temp_now_points[i].x != 0)
                {
                    x_temp_sum = temp_now_points[i].x - pre_kp[i].x;
                    y_temp_sum = temp_now_points[i].y - pre_kp[i].y;
                    num_count_temp++;
                    cout << "num_count_temp = " << num_count_temp << endl;
                }
            }

            x_temp_sum = x_temp_sum / num_count_temp;
            y_temp_sum = y_temp_sum / num_count_temp; //此为检测出的特征点的位置增量的平均值

            for(int i = 0; i < pre_kp.size(); i++)
            {
                if(temp_now_points[i].x == 0)
                {
                    temp_now_points[i].x = pre_kp[i].x + x_temp_sum;
                    temp_now_points[i].y = pre_kp[i].y + y_temp_sum;
                }
            }

            now_points = temp_now_points;

            vector<Point>().swap(temp_now_points);
        }

        //下一时刻的雅可比矩阵估计
        //估计下一时刻的H_k
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
        //估计下一时刻的y_k
        int y_k_count = 0;
        for(int i = 0; i < N; i++)
        {
            y_k(y_k_count, 0) = now_points[i].x - pre_kp[i].x;
            y_k(y_k_count + 1, 0) = now_points[i].y - pre_kp[i].y;
            y_k_count += 2;
        }
        Y_k.push_back(y_k);

//        //kalman滤波核心公式
//        //kalman滤波更新图像雅可比
//        x_k_ = A * X_k[loop_count - 1];
//        X_k_.push_back(x_k_);
//        //预测误差协方差矩阵
//        p_k_ = A * P_k[loop_count - 1] * A.transpose() + Q;
//        P_k_.push_back(p_k_);
//        //开始更新
//        k_k = P_k_[loop_count] * H_k[loop_count - 1].transpose() * (H_k[loop_count - 1] * P_k_[loop_count] * H_k[loop_count - 1].transpose() + R).inverse();
//        K_k.push_back(k_k);
//        //估计值
//        x_k = X_k_[loop_count] + K_k[loop_count] * (Y_k[loop_count] - H_k[loop_count - 1] * X_k_[loop_count]);
//        X_k.push_back(x_k);
//        //估计误差协方差矩阵
//        p_k = (MatrixXd::Identity(2 * N * 6, 2 * N * 6) - K_k[loop_count] * H_k[loop_count - 1]) * P_k_[loop_count];
//        P_k.push_back(p_k);

        pre_joint_pos = now_joint_pos;


        //利用broyden更新雅可比
        MatrixXd delta_joint_pos = now_joint_pos - pre_joint_pos;
        MatrixXd delta_image_pos = y_k;
        MatrixXd delta_temp =  delta_joint_pos.transpose() * pre_rsl * delta_joint_pos;
        MatrixXd now_rsl = 1 / lambda * (pre_rsl - (pre_rsl * delta_joint_pos * delta_joint_pos.transpose() * pre_rsl) / (lambda + delta_temp(0, 0)));
        img_jacobi = pre_jacobian + (delta_image_pos - pre_jacobian * delta_joint_pos) * delta_joint_pos.transpose() * pre_rsl / (lambda + delta_temp(0, 0));

        pre_jacobian = img_jacobi;
        pre_rsl = now_rsl;

        loop_count++;
    }
    return 0;
}
