// telecalibrate.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define _CRT_SECURE_NO_WARNINGS
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"
#include "SnavelyReprojectionError2.h"
#include "glog/logging.h"
#include <iostream>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <eigen3/Eigen/QR>
#include <Python.h>
#include <chrono>
#include <thread>
#include <ctime>
//#include "matplotlibcpp.h"
//#include <cmath>
//namespace plt = matplotlibcpp;



#define image_count 16
#define w_sq 1.5//mm
#define point_num 81 //12*9=108
using namespace Eigen;
using namespace std;
using namespace cv;
using namespace std::chrono;

void Firstoptimize(BALProblem& bal_problem) {
    
    const int point_block_size = bal_problem.point_block_size();//3
    const int camera_block_size = bal_problem.camera_block_size();//6
    const int inparameter_block_size = bal_problem.inparameter_block_size();//9
    double* points = bal_problem.mutable_points();
    double* cameras = bal_problem.mutable_cameras();
    double* inparameters = bal_problem.mutable_inparameters();


//     double *reprojectionerrors_u = bal_problem.mutable_reprojectionerrors_u();
//     double *reprojectionerrors_v = bal_problem.mutable_reprojectionerrors_v();
    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double* observations = bal_problem.observations();//数组observation_
    const double* constpoints = bal_problem.constpoint();



    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i) {

        ceres::CostFunction* cost_function;
        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1], constpoints[3 * bal_problem.point_index()[i] + 0], constpoints[3 * bal_problem.point_index()[i] + 1], constpoints[3 * bal_problem.point_index()[i] + 2]);

        // If enabled use Huber's loss function.
        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.

        double* camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double* point = points + point_block_size * bal_problem.point_index()[i];
        double* inparameter = inparameters;


 //         double *reprojectionerror_u = reprojectionerrors_u;
 //         double *reprojectionerror_v = reprojectionerrors_v;

        problem.AddResidualBlock(cost_function, loss_function, camera, point, inparameter);//loss_function
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
        << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;
    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;//最小化器类型
    options.gradient_tolerance = 1e-10;
    options.function_tolerance = 1e-10;
    options.max_num_consecutive_invalid_steps = 5;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.initial_trust_region_radius = 1e4;
    options.max_trust_region_radius = 1e16;
    options.min_trust_region_radius = 1e-32;
    options.max_num_iterations = 200;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;//SPARSE_SCHUR
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
}



void Secondoptimize(BALProblem& bal_problem) {


    /// 更新世界坐标
    bal_problem.UpdatePoint();

    const int point_block_size = bal_problem.point_block_size();//3
    const int camera_block_size = bal_problem.camera_block_size();//6
    const int inparameter_block_size = bal_problem.inparameter_block_size();//9
    //double* points = bal_problem.mutable_points();
    double* cameras = bal_problem.mutable_cameras();
    double* inparameters = bal_problem.mutable_inparameters();


    //     double *reprojectionerrors_u = bal_problem.mutable_reprojectionerrors_u();
    //     double *reprojectionerrors_v = bal_problem.mutable_reprojectionerrors_v();
        // Observations is 2 * num_observations long array observations
        // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
        // and y position of the observation.
    const double* observations = bal_problem.observations();//数组observation_
    const double* constpoints = bal_problem.constpoint();

    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i) {

        ceres::CostFunction* cost_function;
        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError2::Create(observations[2 * i + 0], observations[2 * i + 1], constpoints[3 * bal_problem.point_index()[i] + 0], constpoints[3 * bal_problem.point_index()[i] + 1], constpoints[3 * bal_problem.point_index()[i] + 2]);

        // If enabled use Huber's loss function.
        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.

        double* camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        //double* point = points + point_block_size * bal_problem.point_index()[i];
        double* inparameter = inparameters;


        //         double *reprojectionerror_u = reprojectionerrors_u;
        //         double *reprojectionerror_v = reprojectionerrors_v;

        problem.AddResidualBlock(cost_function, loss_function, camera, inparameter);//loss_function
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
        << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;//最小化器类型
    options.gradient_tolerance = 1e-10;
    options.function_tolerance = 1e-10;
    options.max_num_consecutive_invalid_steps = 5;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.initial_trust_region_radius = 1e4;
    options.max_trust_region_radius = 1e16;
    options.min_trust_region_radius = 1e-32;
    options.max_num_iterations = 200;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;//SPARSE_SCHUR
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
}



void estimate_params(const std::string& root) {
    int w, h;
    w = 9;
    h = 9;
    int num_point = w * h;
    Size pattern_size = Size(w, h);
    Mat srcImage, grayImage;
    string filename, path, imname;
    double m;
    path = root;

    ///world points
    vector<Point2f> object_points;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            Point2f realPoint;
            realPoint.x = j * w_sq;
            realPoint.y = i * w_sq;
            object_points.push_back(realPoint);
        }
    }

    Matrix<double, Dynamic, Dynamic> M;
    M.setZero(162, 162);  //108*2=216                         
    for (int p = 0, k = 0; k < 2 * num_point; k += 2) {
        M(k, 0) = object_points[p].x;
        M(k, 1) = object_points[p].y;
        M(k, 2) = 1;
        M(k + 1.0, 3) = object_points[p].x;
        M(k + 1.0, 4) = object_points[p].y;
        M(k + 1.0, 5) = 1;
        p++;
    }
    float u0;
    float v0;
    vector<Matrix<double, 3, 3>> homographies(image_count, Matrix<double, 3, 3>(3, 3));
    vector<Matrix<double, 3, 3>> k_mats(image_count, Matrix<double, 3, 3>(3, 3));
    vector<Matrix<double, 2 * point_num, 1>> corner(image_count, Matrix<double, 2 * point_num, 1>(2 * point_num, 1));

    ///detect point each image
    for (int i = 0; i < image_count; i++) {//image_count
        filename = path + "cb0_" + to_string(i) + ".jpg";
        srcImage = imread(filename);
        imname = "img" + to_string(i);
        u0 = srcImage.cols / 2;
        v0 = srcImage.rows / 2;
        vector<Point2f> corners;
        Mat gray;
        cvtColor(srcImage, gray, COLOR_BGR2GRAY);
        int rows = srcImage.rows;
        int cols = srcImage.cols;

        //points detector
        SimpleBlobDetector::Params params;//change the params config according the color of blob
//         params.minThreshold = 0;
//         params.maxThreshold = 150;
//         params.filterByArea = true;
//         params.minArea = 50;
//         params.maxArea = 10e6;
//         params.minDistBetweenBlobs = 10;
        params.filterByColor = true;
        params.blobColor = 0;
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
        bool found = findCirclesGrid(gray, pattern_size, corners, CALIB_CB_SYMMETRIC_GRID, detector);
        //drawChessboardCorners(srcImage, pattern_size, corners, found);
        //imshow(imname,srcImage);
        //waitKey(100);
        //if(i==1) imwrite("1.jpg", srcImage);
        cout << "image_" << to_string(i) << "_:" << found << endl;

        //load imgpoints
        //cout << "corners = \n" << corners << endl;
        //cout << "corners(0,0) = " << corners[80].x << endl;
        for (int j = 0,flag = 0; flag < point_num; flag++) {
            corner[i](j, 0) = corners[flag].x;
            corner[i](j + 1, 0) = corners[flag].y;
            j = j + 2;
        }
        //cout << "修改后 ： \n" << corner[i] << endl;

        Matrix<double, 162, 1> h = M.colPivHouseholderQr().solve(corner[i]); //以前216=108*2
        homographies[i](0, 0) = h(0, 0);
        homographies[i](0, 1) = h(1, 0);
        homographies[i](0, 2) = h(2, 0);
        homographies[i](1, 0) = h(3, 0);
        homographies[i](1, 1) = h(4, 0);
        homographies[i](1, 2) = h(5, 0);
        homographies[i](2, 0) = 0;
        homographies[i](2, 1) = 0;
        homographies[i](2, 2) = 1;
        double c1 = h(0, 0) * h(0, 0) + h(1, 0) * h(1, 0) + h(3, 0) * h(3, 0) + h(4, 0) * h(4, 0);
        double c2 = (h(0, 0) * h(4, 0) - h(1, 0) * h(3, 0)) * (h(0, 0) * h(4, 0) - h(1, 0) * h(3, 0));
        Mat coef = (Mat_<double>(5, 1) << c2, 0, -c1, 0, 1);
        Mat roots;
        solvePoly(coef, roots);
        vector<double> coff = { roots.at<double>(0,0),roots.at<double>(1,0),roots.at<double>(2,0),roots.at<double>(3,0) };
        m = *max_element(coff.begin(), coff.end());
        k_mats[i] << m, 0, u0, 0, m, v0, 0, 0, 1;      //internal matrix each picture
    }

    //compute globle internal matrix
    Matrix<double, image_count, image_count> G;
    G.setOnes(image_count, image_count);
    Matrix<double, image_count, 1> W;
    W.setZero(image_count, 1);
    for (int k = 0; k < image_count; k++) {
        Matrix<double, 3, 3> H = homographies[k];
        G(k, 1) = -(H(1, 0) * H(1, 0)) - (H(1, 1) * H(1, 1));
        G(k, 2) = -(H(0, 0) * H(0, 0)) - (H(0, 1) * H(0, 1));
        G(k, 3) = 2 * (H(0, 0) * H(1, 0) + H(0, 1) * H(1, 1));
        W(k, 0) = -((H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0)) * (H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0)));
    }

    Matrix<double, image_count, 1> L = G.colPivHouseholderQr().solve(W);//colPivHouseholderQr()

    double alpha = sqrt((L(1, 0) * L(2, 0) - L(3, 0) * L(3, 0)) / L(2, 0));
    double beta = sqrt(L(2, 0));
    double gama = sqrt(L(1, 0) - (L(0, 0) / L(2, 0)));
    //     cout << "fx =  " << alpha << "\n" << "fy =  " << beta << endl;

    Matrix<double, 3, 3> K_lsp;
    K_lsp << alpha, 0, u0, 0, beta, v0, 0, 0, 1;
    cout << "Internal Matrix = \n" << K_lsp << endl;

    // compute external matrix
    Vector3d T;//Vector3d--eigen
    vector<double> R_vec;
    Mat R;
    double even_error_u = 0;
    double even_error_v = 0;

    vector<vector<double>> Rs;
    vector<Vector3d> Ts;

    for (int k = 0; k < image_count; k++) {
        Matrix<double, 3, 3> H = homographies[k];
        Matrix<double, 3, 3> K = k_mats[k];
        Matrix<double, 3, 3> E = K.colPivHouseholderQr().solve(H);
        double tx = (H(0, 2) - u0) / m;
        double ty = (H(1, 2) - v0) / m;
        double r13 = sqrt(1 - E(0, 0) * E(0, 0) - E(1, 0) * E(1, 0));
        double r23 = sqrt(1 - E(0, 1) * E(0, 1) - E(1, 1) * E(1, 1));
        Vector3d r1(E(0, 0), E(0, 1), r13);
        Vector3d r2(E(1, 0), E(1, 1), r23);
        Vector3d r3 = r1.cross(r2);
        T << tx, ty, 0;
        Matrix3d Rm;
        Rm << r1, r2, r3;
        cout << "***********file__" << to_string(k + 1) << "__*************" << endl;
        R = (Mat_<double>(3, 3) << E(0, 0), E(0, 1), r13,
            E(1, 0), E(1, 1), r23,
            r3(0, 0), r3(1, 0), r3(2, 0));
   
        Rodrigues(R, R_vec);//opencv--Mat vector<double>

        Rs.push_back(R_vec);
        Ts.push_back(T);

        vector<double> e_u(point_num);
        vector<double> e_v(point_num);
        double pic_error_u = 0;
        double pic_error_v = 0;
    }
    destroyAllWindows();



    ////////write params to file////////////////////////////////////////////
    string filepath = path + "prefinal.txt";
    FILE* fptr = fopen(filepath.c_str(), "w");
    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filepath;
        return;
    }
    //image_num-point_num-image*point_num
    fprintf(fptr, "%d %d %d\n", image_count, point_num, image_count * point_num);
    //observation data
    for (int i = 0; i < image_count; i++)
        for (int flag = 0, j = 0; flag < point_num; flag++, j += 2)
            fprintf(fptr, "%d %d %.6f %.6f\n", i, flag, corner[i](j, 0), corner[i](j + 1.0, 0));

    //     //external data each picture
    //     for(int i = 0; i < Rs.size(); i++){
    //         for(int j = 0; j < Rs[i].size(); j++)
    //             fprintf(fptr, "%.6f\n", Rs[i][j]);//Rs[i][j]
    //         for(int j = 0; j < Rs[i].size(); j++)
    //             fprintf(fptr, "%.6f\n", Ts[i][j]);
    //     }

    for (int i = 0; i < Rs.size(); i++) {
        for (int j = 0; j < Rs[i].size(); j++) {
            fprintf(fptr, "%.6f ", Rs[i][j]);
        }
        fprintf(fptr, "\n");
    }

    for (int i = 0; i < Rs.size(); i++) {
        for (int j = 0; j < Rs[i].size(); j++) {
            fprintf(fptr, "%.6f ", Ts[i][j]);
        }
        fprintf(fptr, "\n");
    }

    //     //world points
    for (int i = 0; i < point_num; i++)
    {
        fprintf(fptr, "%.6f ", object_points[i].y);//如果是输出paramsfile.txt就先输出y,后输出x
        fprintf(fptr, "%.6f ", object_points[i].x);
        fprintf(fptr, "%d\n", 0);
    }
    //     //internal param of camera-----fx,fy,u0,v0,k1,k2,p1,p2,k3
    fprintf(fptr, "%.6f\n", alpha);
    fprintf(fptr, "%.6f\n", beta);
    fprintf(fptr, "%.6f\n", u0);
    fprintf(fptr, "%.6f\n", v0);
    for (int i = 0; i < 5; i++)//initial distortion as 0
        fprintf(fptr, "%d\n", 0);

    fclose(fptr);
    ////////write params to file////////////////////////////////////////////
}




int main(int argc, char** argv)
{
    /////时间戳配置
    // Step one: 定义一个clock
    typedef system_clock sys_clk_t;
    // Step two: 分别获取两个时刻的时间
    typedef system_clock::time_point time_point_t;
    typedef duration<int, std::ratio<1, 1000>> mili_sec_t;

    google::InitGoogleLogging(argv[0]);
    string rootpath = "D:/xwf/Pictures/0912xwf/calibration_target/";//图片存储位置
    /////////estimate_params(rootpath);

    /////////string path_txt = rootpath + "paramsfile.txt";
    BALProblem bal_problem(rootpath, 0);//0--don`t use quaternions
    //bal_problem.WriteToFile("F:/Shaw/projects/VS_Projects/twostepoptimize/circleL/initial.txt");
    bal_problem.ReprojectError(bal_problem);
    //开始时间
    time_point_t start = sys_clk_t::now();
    Firstoptimize(bal_problem);
    //cout << "......FirstOptimized......." << endl;
    //bal_problem.ReprojectError(bal_problem);
    //bal_problem.WriteToFile("F:/Shaw/projects/VS_Projects/twostepoptimize/circleL/firstoptimized.txt");

    Secondoptimize(bal_problem);
    time_point_t end = sys_clk_t::now();
    cout << "time_cost = " << (time_point_cast<mili_sec_t>(end) - time_point_cast<mili_sec_t>(start)).count() << endl;
    cout << "......SecondOptimized......." << endl;
    bal_problem.ReprojectError(bal_problem);
    bal_problem.WriteToFile("D:/xwf/Pictures/0912xwf/Additional Experiments/TwoStepOptimized.txt");
    return 0;
}


// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
