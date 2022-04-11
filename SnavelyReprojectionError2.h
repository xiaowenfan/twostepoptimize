#pragma once
#ifndef SnavelyReprojection2_H
#define SnavelyReprojection2_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError2 {
public:
    SnavelyReprojectionError2(double observation_x, double observation_y, double constpoint_x, double constpoint_y, double constpoint_z) : observed_x(observation_x), observed_y(observation_y), conpoint_x(constpoint_x), conpoint_y(constpoint_y), conpoint_z(constpoint_z) {}

    
    template<typename T>
    bool operator()(const T* const camera,//3����ת����+3��ƽ������
        const T* const inparameter,//9���ڲΣ�fx,fy,ux,uy,k1,k2,p1,p2,k3
        T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];

        T point[3] = { T(conpoint_x), T(conpoint_y), T(conpoint_z) };

        CamProjectionWithDistortion(camera, point, inparameter, predictions);

        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
        //residuals[2] = point[0] * conpoint_y - point[1] * conpoint_x;   //�����Ż�û��Լ������
        //residuals[3] = point[0] - conpoint_x;
        //residuals[4] = point[1] - conpoint_y;
        //residuals[5] = point[2] - conpoint_z;

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T* camera, const T* point, const T* inparameter, T* predictions) {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p);//��ת����--->��ת����
        // camera[3,4,5] are the translation
        p[0] += camera[3];    //Xc
        p[1] += camera[4];//P = R*X+t     R����ת����   Yc
        p[2] += camera[5];    //Zc        
//         std::cout<<"ͼ������ϵ�µ�Z�����꣺"��<<p[0]<<"\n";
        // Compute the center fo distortion

//         T xp = p[0]/p[2];//p = P/P.z   ��һ������ 
//         T yp = p[1]/p[2];                                           // ע����Զ�����������

        // Apply second and fourth order radial distortion
        const T& l1 = inparameter[4];  //��һ���Ż����������
        const T& l2 = inparameter[5];
        const T& l3 = inparameter[8];
        const T& p1 = inparameter[6];
        const T& p2 = inparameter[7];
        const T& focalx = inparameter[0];
        const T& focaly = inparameter[1];
        const T& cx = inparameter[2];
        const T& cy = inparameter[3];
        //         T du = p[0] - cx;
        //         T dv = p[1] - cy;

        T r2 = p[0] * p[0] + p[1] * p[1];  //  xp * xp + yp * yp;               //Զ�����������  du==p[0]   dv == p[1]

        T distortion = T(1.0) + r2 * (l1 + r2 * (l2 + l3 * r2));//add the k3
//         T distortion =  T(1.0) + r2 * ( l1 + r2 * l2);
        T xd = p[0] * distortion;// +T(2.0) * p1 * p[0] * p[1] + p2 * (r2 + T(2.0) * p[0] * p[0]);// +p[0] * (p1 * p[0] - p2 * p[1]);//Զ�����������  xp * distortion + T(2.0) * p1 * xp * yp + p2 * (r2 +T(2) * xp * xp);du
        T yd = p[1] * distortion;// +p1 * (r2 + T(2.0) * p[1] * p[1]) + T(2.0) * p2 * p[0] * p[1];// +p[1] * (p1 * p[0] - p2 * p[1]);//Զ�����������  yp * distortion + p1 * (r2 + T(2.0) * yp * yp) + T(2) * p2 * xp *yp;

        ////modify code according matlab code
//         T xd = l1 * p[0] * r2 + p1 * p[0] * p[0] + p2 * p[0] * p[1] + l2 * r2;
//         T yd = l1 * p[1] * r2 + p2 * p[1] * p[1] + p1 * p[0] * p[1] + l3 * r2;


        predictions[0] = focalx * xd + cx;//��������䣬����ֱ����p[0],p[1]
        predictions[1] = focaly * yd + cy;
        //std::cout << "predictions[0] = " << predictions[0] << "   predictions[1] = " << predictions[1] << std::endl;

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double conpoint_x, const double conpoint_y, const double conpoint_z) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError2, 2, 6, 9>(//9���ڲΣ�3��ʾһ���������xyz����ֵ��6��ʾһ�������3����ת����+3��ƽ����������һ��6��ʾ��6��residuals
            new SnavelyReprojectionError2(observed_x, observed_y, conpoint_x, conpoint_y, conpoint_z)));
    }///////////////////////////////////

private:
    double observed_x;
    double observed_y;
    double conpoint_x;
    double conpoint_y;
    double conpoint_z;

};

#endif // SnavelyReprojection.h

