//
// Created by ctyou on 18-12-22.
//
//此初始化程序要求给定6次指定的位置
//每次运动后拍照截取图片，计算图像特征点坐标，当前的6个关节的角度
//所以初始化的复合雅可比矩阵为：jac = [ds1, ds2, ds3, ds4, ds5, ds6] * [dq1, dq2, dq3, dq4, dq5, dq6].inverse()
//所以算上机器人初始位置的话，一共需要计算7次图像特征点，7次关节角度

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
//#include "/home/ctyou/ibvs_ros/devel/include/ibvs_core/ibvs_core.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace KDL;

#define N 4

//关键点排序用数据结构
struct kp_stru{
    double cd;     //x坐标
    int num;       //序号
};

bool comp(const kp_stru &kp1, const kp_stru &kp2)
{
    return kp1.cd <= kp2.cd;
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

//    for(int i = 0; i < n; i++)
//        cout << "cd_2 = " << cd_2[i] << endl;
//    for(int i = 0; i < n - 1; i++)
//        cout << "cd_diff = " << cd_diff[i] << endl;

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

//vector<double> joint_position;
double joint_position[6] = {1, 1, 1, 1, 1, 1};

//订阅UR的6个实际角度位置，并计算末端位姿
void joint_position_callback(sensor_msgs::JointState msg)
{
    for(int i = 0; i < 6; i++)
        joint_position[i] = msg.position[i];
}

int main(int argc, char ** argv) {

    ros::init(argc, argv, "init_par_2");
    ros::NodeHandle n_ros;

    ros::Subscriber sub_joint_pos = n_ros.subscribe("joint_states", 100, joint_position_callback);
    ros::Publisher velocity_pub = n_ros.advertise<trajectory_msgs::JointTrajectory>("ur_driver/joint_speed", 100);

    Tree my_tree;
    if (!kdl_parser::treeFromFile("/home/ctyou/ibvs_ros/src/ur/universal_robot-kinetic-devel/ur_description/urdf/ur3_robot.urdf", my_tree)) {
        ROS_ERROR("Failed to construct kdl tree");
    }

    bool exit_value;
    Chain my_chain;
    exit_value = my_tree.getChain("base_link", "ee_link", my_chain);
    if (!exit_value)
        ROS_ERROR("Failed to convert to chain");

    Size sz = Size(640, 480);
    //检测椭圆相关准备
    // Parameters Settings (Sect. 4.2)
    int	    iThLength = 16;
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

    //图像有关变量,检测椭圆有关变量
    vector<Mat> src(7);
    vector<Mat> src_l(7);
    vector<Mat1b> src_l_gray(7);
    vector<Mat3b> draw_src(7);
    vector< vector<Ellipse> > ellsYaed(7);
    vector< vector<Point> > points(7);

    MatrixXd c = MatrixXd::Zero(6, 1);
    vector<MatrixXd> joint_value(7, c);

    for(int num_count = 0; num_count < 7; num_count++)
    {

        if(num_count > 0)
        {
            cout << "开始第 " << num_count << " 次的收集，请移动机械臂！移动完毕后请按任意数字键！" << endl;
            int temp_pause_2;
            cin >> temp_pause_2;
        }

        while(1) {
            cap >> src[num_count];
            if(src[num_count].empty())
            {
                cout << "摄像头未能输入图像！" << endl;
                return -1;
            }
            src_l[num_count] = src[num_count].colRange(1, 640).clone();
            cvtColor(src_l[num_count], src_l_gray[num_count], CV_BGR2GRAY);
            yaed->Detect(src_l_gray[num_count], ellsYaed[num_count]);
            draw_src[num_count] = src_l[num_count].clone();
            yaed->DrawDetectedEllipses(draw_src[0], ellsYaed[0]);

            imshow("draw_ells", draw_src[num_count]);
            waitKey(5);
            if(ellsYaed[num_count].size() < 4)
            {
                cout << "该帧少于4个特征点!" << endl;
                cout << "现在请自行移动机器人！移动完毕后请按任意数字键盘！注意不要和上次的位置类似！" << endl;
                int temp_pause_2;
                cin >> temp_pause_2;
                continue;
            }

            for (int i = 0; i < ellsYaed[num_count].size(); i++) {
                points[num_count].push_back(Point(ellsYaed[num_count][i]._xc, ellsYaed[num_count][i]._yc));
            }
            points[num_count] = sort_keypoints(points[num_count]);

            if(points[num_count].size() != 4)
            {
                cout << "该帧不能检测到4个特征点!" << endl;
                cout << "现在请自行移动机器人！移动完毕后请按任意数字键盘！注意不要和上次的位置类似！" << endl;
                vector<Point>().swap(points[num_count]);
                vector<Ellipse>().swap(ellsYaed[num_count]);
                int temp_pause_3;
                cin >> temp_pause_3;
            }
            else {
                cout << "已经检测到4个特征点，开始下一步" << endl;
                cout << "  " << endl;
                break;
            }
        }

        cout << "收集此时刻的关节角度值..." << endl;
        ros::spinOnce();
        for (int i = 0; i < 6; i++)
            joint_value[num_count](i, 0) = joint_position[i];
    }

    cout << " " << endl;
    cout << "移动完毕，开始计算复合雅可比！" << endl;

    MatrixXd d_s(2 * N, 6);
    MatrixXd a_temp = MatrixXd::Zero(2 * N, 1);
    vector<MatrixXd> d_s_b(6, a_temp);
    MatrixXd d_q(6, 6);
    MatrixXd jac_complex(2 * N, 6);

    cout << "yes 1" << endl;

    int point_count = 0;
    for(int i = 0; i < 6; i++) {
        cout << "i = " << i << endl;
        for(int j = 0; j < N; j++) {
                cout << "j = " << j << endl;
                cout << "d_s_b[i](point_count, 0) = " << d_s_b[i](point_count, 0) << endl;
                cout << "points[i + 1][j] = " << points[i + 1][j] << endl;

                d_s_b[i](point_count, 0) = points[i + 1][j].x - points[i][j].x;  //或者试试img_velocity[i]
                d_s_b[i](point_count + 1, 0) = points[i + 1][j].y - points[i][j].y;  //或者试试img_velocity[i]
                point_count = point_count + 2;
                cout << "yes 2" << endl;
        }
        point_count = 0;
    }

    for(int i = 0; i < 6; i++)
        d_s.block< 2 * N, 1 >(0, i) = d_s_b[i];

    cout << "yes 3" << endl;
    for(int i = 0; i < 6; i++) {
        d_q.block<6, 1>(0, i) = joint_value[i + 1] - joint_value[i];
    }

    cout << "d_q = " << endl;
    cout << d_q << endl;

    jac_complex = d_s * d_q.inverse();

    cout << "初始化的复合雅可比矩阵为：" << endl;
    cout << jac_complex << endl;

    return 0;
}

