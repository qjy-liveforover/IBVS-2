//此初始化程序要求给定6次指定的位置
//每次运动后拍照截取图片，计算图像特征点坐标，当前的6个关节的角度
//所以初始化的复合雅可比矩阵为：jac = [ds1, ds2, ds3, ds4, ds5, ds6] * [dq1, dq2, dq3, dq4, dq5, dq6].inverse()
//所以算上机器人初始位置的话，一共需要计算7次图像特征点，7次关节角度

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
using namespace cv;
using namespace Eigen;
using namespace KDL;

#define N 10

//vector<double> joint_position;
double joint_position[6] = {1, 1, 1, 1, 1, 1};

//订阅UR的6个实际角度位置，并计算末端位姿
void joint_position_callback(sensor_msgs::JointState msg)
{
    for(int i = 0; i < 6; i++)
        joint_position[i] = msg.position[i];
}

int main(int argc, char ** argv) {

    ros::init(argc, argv, "init_par");
    ros::NodeHandle n_ros;

    ros::Subscriber sub_joint_pos = n_ros.subscribe("joint_states", 100, joint_position_callback);
    ros::Publisher velocity_pub = n_ros.advertise<trajectory_msgs::JointTrajectory>("ur_driver/joint_speed", 100);

    Tree my_tree;
    if (!kdl_parser::treeFromFile(
            "/home/ctyou/ibvs_ros/src/ur/universal_robot-kinetic-devel/ur_description/urdf/ur3_robot.urdf", my_tree)) {
        ROS_ERROR("Failed to construct kdl tree");
    }

    bool exit_value;
    Chain my_chain;
    exit_value = my_tree.getChain("base_link", "ee_link", my_chain);
    if (!exit_value)
        ROS_ERROR("Failed to convert to chain");

    /*摄像头需要做的事情*/
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

    //图像有关变量
    vector<Mat> src(7);
    vector<Mat> src_l(7);
    Ptr<ORB> orb = ORB::create(N, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    double min_dist = 1000, max_dist = 0;

    ChainFkSolverPos_recursive fksolver = ChainFkSolverPos_recursive(my_chain);    //正解
    unsigned int nj = my_chain.getNrOfJoints();
    int kinematics_status;

    vector<MatrixXd::Zero(6, 1)> joint_value(7);

    //记录初始位置的图像和关节角度（角度为弧度）
    cap >> src(0);
    ros::spinOnce();
    for (int i = 0; i < 6; i++)
        joint_value(0)(i, 0) = joint_position[i];

    //大循环一共循环6次来记录图像和关节角度
    for (int num_count = 1; num_count < 7; num_count++) {
        //机械臂有关变量
        Matrix3d ur_rot;
        Vector3d eal;
        double current_pos[6] = {0, 0, 0, 0, 0, 0};
        JntArray jointpositions = JntArray(nj);
        Frame cartpos;

        //首先计算当前位置和姿态，以便操作者判断下一个目标位置
        ros::spinOnce();
        for (unsigned int i = 0; i < nj; i++)
            jointpositions(i) = joint_position[i];                                  //6个角度实际位置

        kinematics_status = fksolver.JntToCart(jointpositions, cartpos, -1);
        ur_rot(0, 0) = cartpos(0, 0);
        ur_rot(0, 1) = cartpos(0, 1);
        ur_rot(0, 2) = cartpos(0, 2);
        ur_rot(1, 0) = cartpos(1, 0);
        ur_rot(1, 1) = cartpos(1, 1);
        ur_rot(1, 2) = cartpos(1, 2);
        ur_rot(2, 0) = cartpos(2, 0);
        ur_rot(2, 1) = cartpos(2, 1);
        ur_rot(2, 2) = cartpos(2, 2);

        eal = ur_rot.eulerAngles(2, 1, 0);  //转换成欧拉角

        current_pos[0] = cartpos(0, 3);
        current_pos[1] = cartpos(1, 3);
        current_pos[2] = cartpos(2, 3);
        current_pos[3] = eal(0, 0);
        current_pos[4] = eal(1, 0);
        current_pos[5] = eal(2, 0);

        cout << "机器人现在的位置为：" << "X: " << current_pos[0] << " Y: " << current_pos[1] << " Z: " << current_pos[2]
             << " yaw "
             << current_pos[3] << " pitch " << current_pos[4] << " roll " << current_pos[5] << endl;
        double desired_x, desired_y, desired_z, desired_yaw, desired_pitch, desired_roll;

        vector<double> desired_pos;
        cout << "请输入机器人末端的期望位姿: x y z yaw pitch roll" << endl;
        cin >> desired_x >> desired_y >> desired_z >> desired_yaw >> desired_pitch >> desired_roll;

        desired_pos.push_back(desired_x);
        desired_pos.push_back(desired_y);
        desired_pos.push_back(desired_z);
        desired_pos.push_back(desired_yaw);
        desired_pos.push_back(desired_pitch);
        desired_pos.push_back(desired_roll);

        VectorXd a(6);
        vector<VectorXd> e_con(3, a);
        VectorXd cart_velocity(6);

        ros::Rate loop_rate(100);
        int count_i = 0;
        int count_pid = 0;
        ofstream outfile;

        while (ros::ok()) {

            ros::spinOnce();
            loop_rate.sleep();

            for (unsigned int i = 0; i < nj; i++)
                jointpositions(i) = joint_position[i];//6个角度实际位置

            kinematics_status = fksolver.JntToCart(jointpositions, cartpos, -1);    //FK里面的参数是6个关节的角度，输出的是4×4位姿矩阵。

            ur_rot(0, 0) = cartpos(0, 0);
            ur_rot(0, 1) = cartpos(0, 1);
            ur_rot(0, 2) = cartpos(0, 2);
            ur_rot(1, 0) = cartpos(1, 0);
            ur_rot(1, 1) = cartpos(1, 1);
            ur_rot(1, 2) = cartpos(1, 2);
            ur_rot(2, 0) = cartpos(2, 0);
            ur_rot(2, 1) = cartpos(2, 1);
            ur_rot(2, 2) = cartpos(2, 2);

            eal = ur_rot.eulerAngles(2, 1, 0);

            current_pos[0] = cartpos(0, 3);
            current_pos[1] = cartpos(1, 3);
            current_pos[2] = cartpos(2, 3);
            current_pos[3] = eal(0, 0);
            current_pos[4] = eal(1, 0);
            current_pos[5] = eal(2, 0);

/*根据上面算出的当前的位置和姿态，求解每个关节应该发送的速度, 运用增量式PID求速度*/
            double p_k = 0.01, i_k = 0.001, d_k = 0.01;
            VectorXd e(6);
            VectorXd vel_delta(6);

            e[0] = (desired_pos[0] - current_pos[0]);
            e[1] = (desired_pos[1] - current_pos[1]);
            e[2] = (desired_pos[2] - current_pos[2]);
            e[3] = (desired_pos[3] - current_pos[3]);
            e[4] = (desired_pos[4] - current_pos[4]);
            e[5] = (desired_pos[5] - current_pos[5]);

            e_con[0] = e;

            if (count_pid == 0) {
                e_con[1] = e;
                e_con[2] = e;

                cart_velocity[0] = p_k * e_con[0][0];
                cart_velocity[1] = p_k * e_con[0][1];
                cart_velocity[2] = p_k * e_con[0][2];
                cart_velocity[3] = p_k * e_con[0][3];
                cart_velocity[4] = p_k * e_con[0][4];
                cart_velocity[5] = p_k * e_con[0][5];
            } else {
                vel_delta[0] = p_k * (e_con[0][0] - e_con[1][0]) + i_k * e_con[0][0] +
                               d_k * (e_con[0][0] - 2 * e_con[1][0] + e_con[2][0]);
                vel_delta[1] = p_k * (e_con[0][1] - e_con[1][1]) + i_k * e_con[0][1] +
                               d_k * (e_con[0][1] - 2 * e_con[1][1] + e_con[2][1]);
                vel_delta[2] = p_k * (e_con[0][2] - e_con[1][2]) + i_k * e_con[0][2] +
                               d_k * (e_con[0][2] - 2 * e_con[1][2] + e_con[2][2]);
                vel_delta[3] = p_k * (e_con[0][3] - e_con[1][3]) + i_k * e_con[0][3] +
                               d_k * (e_con[0][3] - 2 * e_con[1][3] + e_con[2][3]);
                vel_delta[4] = p_k * (e_con[0][4] - e_con[1][4]) + i_k * e_con[0][4] +
                               d_k * (e_con[0][4] - 2 * e_con[1][4] + e_con[2][4]);
                vel_delta[5] = p_k * (e_con[0][5] - e_con[1][5]) + i_k * e_con[0][5] +
                               d_k * (e_con[0][5] - 2 * e_con[1][5] + e_con[2][5]);

                cart_velocity[0] = cart_velocity[0] + vel_delta[0];
                cart_velocity[1] = cart_velocity[1] + vel_delta[1];
                cart_velocity[2] = cart_velocity[2] + vel_delta[2];
                cart_velocity[3] = cart_velocity[3] + vel_delta[3];
                cart_velocity[4] = cart_velocity[4] + vel_delta[4];
                cart_velocity[5] = cart_velocity[5] + vel_delta[5];
            }

            VectorXd e_tmp;
            e_tmp = e_con[1];
            e_con[1] = e_con[0];
            e_con[2] = e_tmp;

            for (int i = 0; i < 6; i++) {
                cout << "发送的第" << i << "个关节的速度为： " << cart_velocity[i] << endl;
            }

            cout << "与期望坐标的差距： " << endl;
            for (int i = 0; i < 6; i++) {
                cout << (desired_pos[i] - current_pos[i]) << endl;
            }

            if (abs(desired_pos[0] - current_pos[0]) < 0.001 && abs(desired_pos[1] - current_pos[1]) < 0.001 &&
                abs(desired_pos[2] - current_pos[2]) < 0.001)
                break;

            double d[6];
            d[0] = cart_velocity[0];
            d[1] = cart_velocity[1];
            d[2] = cart_velocity[2];
            d[3] = cart_velocity[3];
            d[4] = cart_velocity[4];
            d[5] = cart_velocity[5];

            Twist my_twist;
            my_twist(0) = d[0];
            my_twist(1) = d[1];
            my_twist(2) = d[2];
            my_twist(3) = d[3];
            my_twist(4) = d[4];
            my_twist(5) = d[5];

            JntArray joint_cal_vel = JntArray(nj);
            ChainIkSolverVel_pinv vel_pinv_solver = ChainIkSolverVel_pinv(my_chain, 0.00001, 150);
            vel_pinv_solver.CartToJnt(jointpositions, my_twist, joint_cal_vel);

            //计算出的要发送的速度
            trajectory_msgs::JointTrajectory velocity_msg;
            velocity_msg.joint_names.resize(6);
            velocity_msg.joint_names[0] = "shoulder_pan_joint";
            velocity_msg.joint_names[1] = "shoulder_lift_joint";
            velocity_msg.joint_names[2] = "elbow_joint";
            velocity_msg.joint_names[3] = "wrist_1_joint";
            velocity_msg.joint_names[4] = "wrist_2_joint";
            velocity_msg.joint_names[5] = "wrist_3_joint";
            velocity_msg.header.stamp = ros::Time::now();
            velocity_msg.points.resize(1);
            velocity_msg.points[0].velocities.resize(6);
            velocity_msg.points[0].velocities[0] = joint_cal_vel(0);
            velocity_msg.points[0].velocities[1] = joint_cal_vel(1);
            velocity_msg.points[0].velocities[2] = joint_cal_vel(2);
            velocity_msg.points[0].velocities[3] = joint_cal_vel(3);
            velocity_msg.points[0].velocities[4] = joint_cal_vel(4);
            velocity_msg.points[0].velocities[5] = joint_cal_vel(5);
            velocity_pub.publish(velocity_msg); //机器人开始运动

            count_pid++;
        }

        cout << "第" << num_count << "个位置已经走完，现在开始记录图像和关节角度" << endl;
        cap >> src(i);
        if (src(i).empty()) {
            cout << "src is empty" << endl;
            return -1;
        }

        for (int i = 0; i < 6; i++)
            joint_value(i)(i, 0) = joint_position[i];
    }

    cout << "图像采集完毕，开始计算初始复合雅可比矩阵" << endl;

    vector<KeyPoint> keypoints(7);
    vector<Mat> descriptors(7);

    for (int i = 0; i < 7; i++) {
        orb->detect(src(i), keypoints(i));
        orb->compute(src(i), keypoints(i), descriptors(i));
    }

    vector<vector<DMatch>> matches(6);
    BFMatcher matcher(NORM_HAMMING);
    for (int i = 0; i < 6; i++) {
        matcher.match(descriptors(i), descriptors(i + 1), matches(i));
    }

    VectorXd b(2 * N);
    vector<VectorXd> img_velocity(6, b);

    int point_count = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < matches(0).size(); j++) {
            img_velocity(i)[point_count] = (keypoints(i + 1)[matches(i)[j].trainIdx].pt.x -
                                            keypoints(i)[matches[i].queryIdx].pt.x);
            img_velocity(i)[point_count + 1] = (keypoints(i + 1)[matches(i)[j].trainIdx].pt.y -
                                                keypoints(i)[matches[j].queryIdx].pt.y);
            point_count = point_count + 2;
        }
        point_count = 0;
    }

    MatrixXd d_s(2 * N, 6);
    MatrixXd d_q(6, 6);
    MatrixXd jac_complex(2 * N, 6);

    for (int i = 0; i < 6; i++) {
        d_s < 2 * N, 1 > (0, i) = img_velocity(i).head(2 * N);  //或者试试img_velocity(i)
        d_q < 6, 1 > (0, i) = img_velocity(i) < 6, 1 > (0, 0);
    }

    jac_complex = d_s * d_q.inverse();

    cout << "初始化的复合雅可比矩阵为：" << endl;
    cout << jac_complex << endl;

    return 0;
}
