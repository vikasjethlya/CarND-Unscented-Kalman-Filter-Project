#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

#define SMALL_FLOAT_VAL 0.0001
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /*
  Complete the initialization. See ukf.h for other member properties.

  */

  is_initialized_ = false;
  time_us_ = 0;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  n_z_ = 3;
  nis_laser_ = 0.0 ;
  nis_radar_ = 0.0 ;

  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(0.0);

  x_.fill(1.0);
  P_ << 1,0,0,0,0,
		0,1,0,0,0,
		0,0,1,0,0,
		0,0,0,1,0,
		0,0,0,0,1;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);
  Zsig_.fill(0.0);

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  H_laser_ = MatrixXd(2, 5);

  H_laser_ << 1, 0, 0, 0,0,
    		0, 1, 0, 0,0;

  R_laser_ << std_laspx_ * std_laspx_ , 0,
		  	  0,std_laspy_*std_laspy_;

  R_radar_ = MatrixXd(n_z_,n_z_);
  R_radar_ <<    std_radr_*std_radr_, 0, 0,
           0, std_radphi_*std_radphi_, 0,
           0, 0,std_radrd_*std_radrd_;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	/*****************************************************************************
	   *  Initialization
	 ****************************************************************************/
	  if (!is_initialized_) {

		//define variable to receive RADAR and LIDAR position data

		float px = 0.0;
		float py = 0.0;
		float vx = 0.0;
		float vy = 0.0;

	    // first measurement
	    cout << "EKF: " << endl;

	    //intialize weight

	    SetWeight();

	    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	      /**
	      Convert radar from polar to cartesian coordinates and initialize state.
	      */
				float ro = meas_package.raw_measurements_[0];
				float phi = meas_package.raw_measurements_[1];
				float ro_dot = meas_package.raw_measurements_[2];

				//Normalize Phi
				phi = NormalizePhi(phi);

				px = ro * cos(phi);
				py = ro * sin(phi);
				vx = ro_dot * cos(phi);
				vy = ro_dot * sin(phi);

				x_ << px, py, sqrt(vx*vx + vy*vy),phi,0;

	    }

	    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
	      /**
	      Initialize state.
	      */
			 px = meas_package.raw_measurements_[0];
			 py = meas_package.raw_measurements_[1];
			 x_ << px, py, 0,0,0;
	    }

		//Check for zeros
		  if (fabs(x_(0)) < SMALL_FLOAT_VAL and fabs(x_(1)) < SMALL_FLOAT_VAL){
			  x_(0) = 0.0001;
			  x_(1) = 0.0001;
		  }

	    //add new time stamp from measurment_pack
		  time_us_ = meas_package.timestamp_;


	    // done initializing, no need to predict or update
	    is_initialized_ = true;
	    return;
	  }

	  /*****************************************************************************
	  	 *  Calculate dt
	  	 ****************************************************************************/

	  	float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
	  	time_us_ = meas_package.timestamp_;

	  /*****************************************************************************
	   *  Prediction
	   ****************************************************************************/
	  	// Call predication function

	  	Prediction(dt);

	/*****************************************************************************
	   *  Update
	   ****************************************************************************/

	  /**
		 * Use the sensor type to perform the update step.
		 * Update the state and covariance matrices.
	   */

	  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
		    UpdateRadar(meas_package);

		}
		else
		{
			UpdateLidar(meas_package);
	    }

	  // print the output
	  cout << "x_ = " << x_ << endl;
	  cout << "P_ = " << P_ << endl;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	Xsig_aug.fill(0.0);
	AugmentedSigmaPoints(&Xsig_aug);
	Xsig_pred_ = SigmaPointPrediction(Xsig_aug, delta_t);
	PredictMeanAndCovariance();

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

	float px = meas_package.raw_measurements_[0];
	float py = meas_package.raw_measurements_[1];
	VectorXd z = VectorXd(2);
    z << px,py;

	VectorXd z_pred = H_laser_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_laser_.transpose();
	MatrixXd PHt = P_ * Ht;
	MatrixXd S = H_laser_ * PHt + R_laser_;
	MatrixXd Si = S.inverse();
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_laser_) * P_;

	// calculate NIS

	nis_laser_ = y.transpose()*Si*y;
	cout << "NIS Lidar :" << nis_laser_ << endl;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
	float ro = meas_package.raw_measurements_[0];
	float phi = meas_package.raw_measurements_[1];
	phi = NormalizePhi(phi);
	float ro_dot = meas_package.raw_measurements_[2];

	VectorXd z = VectorXd(n_z_);

	z << ro,phi,ro_dot;

	 //mean predicted measurement
	  VectorXd z_pred = VectorXd(n_z_);
	  MatrixXd S = MatrixXd(n_z_,n_z_);

	  PredictRadarMeasurement(&z_pred , &S);

	  //create matrix for cross correlation Tc
	  MatrixXd Tc = MatrixXd(n_x_, n_z_);


	  //calculate cross correlation matrix
	  Tc.fill(0.0);
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

		//residual
		VectorXd z_diff = Zsig_.col(i) - z_pred;

		//angle normalization
		z_diff(1) = NormalizePhi(z_diff(1));

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		//angle normalization
		x_diff(3) = NormalizePhi(x_diff(3));


		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	  }

	  //Kalman gain K;
	  MatrixXd K = Tc * S.inverse();

	  //residual
	  VectorXd z_diff = z - z_pred;

	  //angle normalization
	  z_diff(1) = NormalizePhi(z_diff(1));

	  //update state mean and covariance matrix
	  x_ = x_ + K * z_diff;
	  P_ = P_ - K*S*K.transpose();

	  //Calculate NIS of Radar

	  nis_radar_ = z_diff.transpose()*S.inverse()*z_diff ;

	  cout << "NIS radar :" << nis_radar_ << endl;

}


/**
 * Create augumented sigma points using mean state, augumented covarience matrix
 * and square root matrix.
 * @param augumented sigma points
 */
void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {


  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);


  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  //print result
  //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  *Xsig_out = Xsig_aug;



}

/**
   * create Predict Sigma points
   * @param augumented sigma points
   * Return : Predicted Sigma point Matrix at delta_t time
 */
MatrixXd UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t) {


  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  //print result
  //std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  return Xsig_pred;

}

/**
     Set weight require in calculation of state mean and covarience matrix
 */
void UKF::SetWeight()
{
	 double weight_0 = lambda_/(lambda_+n_aug_);
	  weights_(0) = weight_0;
	  for (int i=1; i<2*n_aug_+1; i++) {
	    double weight = 0.5/(n_aug_+lambda_);
	    weights_(i) = weight;
	  }
}

/**
	 * Calculate predicted state mean (x) and state covariance matrix (p)

  */

void UKF::PredictMeanAndCovariance() {


  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);



  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x = x+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    //angle normalization
    x_diff(3) = NormalizePhi(x_diff(3));

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }


  //print result
  //std::cout << "Predicted state" << std::endl;
  //std::cout << x << std::endl;
  //std::cout << "Predicted covariance matrix" << std::endl;
  //std::cout << P << std::endl;

  //write result
  x_ = x;
  P_ = P;
}

/**
  	 * Calculate mean predicted measurement and  measurement covariance matrix.
  	 * First need to transform sigma points into measurement space.
  	 * @Param : Mean predicted measurement , Predicted measurement covariance matrix
*/
void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out) {


  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig_(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig_(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig_.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_,n_z_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizePhi(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_radar_;

  //print result
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  *z_out = z_pred;
  *S_out = S;
}

/**
  	 * Normalize yaw angle in range of -PI to PI
  	 * @param : Andgle
  	 * Return : Normalized yaw angle
 */
float UKF::NormalizePhi(float angle){

	if(fabs(angle) > M_PI){
			angle -= round(angle / (2. * M_PI)) * (2.* M_PI);
		}
	return angle;

}

