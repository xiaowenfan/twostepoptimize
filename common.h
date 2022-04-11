#pragma once

/// 从文件读入BAL dataset
class BALProblem {
public:
    /// load bal data from text file
    explicit BALProblem(const std::string &filename,  bool use_quaternions = false);

    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
        delete[] constpoint_;////////////////////////////
        delete[] reprojectu_;
        delete[] reprojectv_;
    }

    
    ///更新世界坐标
    void UpdatePoint();


    /// save results to text file
    void WriteToFile(const std::string &filename) const;

    /// save results to ply pointcloud
    void WriteToPLYFile(const std::string &filename) const;

    ///重投影误差
    void ReprojectError(BALProblem& bal_problem);


    void Normalize();

    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);

    int camera_block_size() const { return use_quaternions_ ? 10 : 6; }//修改成10

    int inparameter_block_size() const { return 9; }///////fx,fy,cx,cy,k1,k2,p1,p2,k3
    
    int point_block_size() const { return 3; }

    int num_cameras() const { return num_cameras_; }

    int num_points() const { return num_points_; }

    int num_observations() const { return num_observations_; }

    int num_parameters() const { return num_parameters_; }

    const int *point_index() const { return point_index_; }

    const int *camera_index() const { return camera_index_; }

    const double *observations() const { return observations_; }
    
    const double *constpoint() const { return constpoint_; }///////////constpoint///////////////

    const double *parameters() const { return parameters_; }

    const double *cameras() const { return parameters_; }

    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }

    /// camera参数的起始地址
    double *mutable_cameras() { return parameters_; }

    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }
   /////////////////////// 
    double *mutable_inparameters() { return parameters_ + camera_block_size() * num_cameras_ + point_block_size() * num_points_;}
    
    double *mutable_reprojectionerrors_u() {return reprojectu_;}//////////////////////////////////////////////////////////////////////
   
    double *mutable_reprojectionerrors_v() {return reprojectv_;}

    double *mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    const double *camera_for_observation(int i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    const double *point_for_observation(int i) const {
        return points() + point_index_[i] * point_block_size();
    }

private:
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    //int num_parametersR_;
    bool use_quaternions_;

    int *point_index_;      // 每个observation对应的point index
    int *camera_index_;     // 每个observation对应的camera index
    double *observations_;
    
    double *parameters_;
    double *constpoint_;////////////////////////////
    double *reprojectu_;
    double *reprojectv_;
};
