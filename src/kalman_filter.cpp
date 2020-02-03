#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {

    // the initial transition matrix F_
    F_ = MatrixXd(4, 4);
    F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

    tools = Tools();
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd& x_in, MatrixXd& P_in, MatrixXd& F_in,
    MatrixXd& H_in, MatrixXd& R_in, MatrixXd& Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    std::cout << "F: " << F_ << std::endl;
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    std::cout << "Ft: " << Ft << std::endl;
    std::cout << "P: " << P_ << std::endl;
    std::cout << "Q: " << Q_ << std::endl;
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd& z) {
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd& z) {
    /**
     * TODO: update the state by using Extended Kalman Filter equations
     */
    std::cout << "z: " << z << std::endl;
    VectorXd z_norm = VectorXd(3);
    z_norm << z(0), tools.NormalizePhi(z(1)), z(2);
    std::cout << "z normalized: " << z_norm << std::endl;
    float c1 = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    VectorXd hx = VectorXd(3);
    hx << c1, atan2(x_(1),x_(0)), (x_(0) * x_(2) + x_(1) * x_(3)) / c1;
    std::cout << "hx: " << hx << std::endl;
    VectorXd y = z - hx;
    VectorXd y_norm = VectorXd(3);
    y_norm << y(0), tools.NormalizePhi(y(1)), y(2);
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y_norm);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
