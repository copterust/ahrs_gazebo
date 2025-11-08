use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ahrs_gazebo::{
    AccelMagMeasurement, GyroscopeProcess, ahrs_state_new, attitude_component, bias_component,
};

use estima::UnscentedKalmanFilter;
use estima::sigma_points::MerweScaledSigmaPoints;
use gz::msgs::{imu, magnetometer};
use gz::transport::Node;
use nalgebra::{Matrix6, U6, UnitQuaternion, Vector3, Vector6};

#[derive(Debug)]
struct SensorData {
    imu: Option<imu::IMU>,
    mag: Option<magnetometer::Magnetometer>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut node = Node::new().ok_or("Failed to create node")?;

    let dt: f64 = 0.01;
    let gyro_noise: f64 = 0.01;
    let accel_noise: f64 = 0.1;
    let mag_noise: f64 = 0.1;

    let initial_state = ahrs_state_new(UnitQuaternion::identity(), Vector3::zeros());
    let initial_covariance = Matrix6::<f64>::identity() * 0.2;

    let mut process_noise = Matrix6::<f64>::zeros();
    process_noise
        .fixed_view_mut::<3, 3>(0, 0)
        .fill_diagonal(0.05f64.powi(2));
    process_noise
        .fixed_view_mut::<3, 3>(3, 3)
        .fill_diagonal(gyro_noise.powi(2) * dt);

    let mut measurement_noise = Matrix6::<f64>::zeros();
    measurement_noise
        .fixed_view_mut::<3, 3>(0, 0)
        .fill_diagonal(accel_noise.powi(2));
    measurement_noise
        .fixed_view_mut::<3, 3>(3, 3)
        .fill_diagonal(mag_noise.powi(2));

    let sigma_gen = MerweScaledSigmaPoints::new(0.5, 2.0, 0.0);
    let weights = sigma_gen.weights::<U6>();

    let measurement_model = AccelMagMeasurement::new();
    let ukf = UnscentedKalmanFilter::new(
        initial_state,
        initial_covariance,
        GyroscopeProcess,
        process_noise,
        measurement_model.clone(),
        measurement_noise,
        sigma_gen,
        weights,
    )
    .with_regularization_factor(1e-3);

    let ukf = Arc::new(Mutex::new(ukf));
    let sensor_data = Arc::new(Mutex::new(SensorData {
        imu: None,
        mag: None,
    }));

    let ukf_clone = ukf.clone();
    let sensor_data_clone = sensor_data.clone();

    if !node.subscribe("/imu", move |imu_msg: imu::IMU| {
        let mut data = match sensor_data_clone.lock() {
            Ok(guard) => guard,
            Err(err) => {
                eprintln!("Recovering from poisoned sensor data mutex");
                err.into_inner()
            }
        };
        data.imu = Some(imu_msg);
        if let (Some(imu), Some(mag)) = (data.imu.as_ref(), data.mag.as_ref()) {
            // Guard against invalid sensor readings (f.e. freefall) that would introduce NaNs into the filter.
            // Think about adding is_valid to trait
            const MIN_SENSOR_NORM_SQ: f64 = 1e-12;

            let linear_acceleration = Vector3::new(
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            );

            if linear_acceleration.norm_squared() < MIN_SENSOR_NORM_SQ {
                eprintln!("Skipping UKF update: accelerometer norm is too small");
                return;
            }

            let mag_field = Vector3::new(mag.field_tesla.x, mag.field_tesla.y, mag.field_tesla.z);

            if mag_field.norm_squared() < MIN_SENSOR_NORM_SQ {
                eprintln!("Skipping UKF update: magnetometer norm is too small");
                return;
            }

            let measurement = Vector6::new(
                linear_acceleration.x,
                linear_acceleration.y,
                linear_acceleration.z,
                mag_field.x,
                mag_field.y,
                mag_field.z,
            );

            drop(data);

            let mut ukf = match ukf_clone.lock() {
                Ok(guard) => guard,
                Err(err) => {
                    eprintln!("Recovering from poisoned UKF mutex");
                    err.into_inner()
                }
            };

            if let Err(err) = ukf.update(&measurement) {
                eprintln!("UKF update failed: {err:?}");
            }
        }
    }) {
        return Err("Failed to subscribe to /imu".into());
    }

    let sensor_data_clone2 = sensor_data.clone();
    if !node.subscribe("/mag", move |mag_msg: magnetometer::Magnetometer| {
        let mut data = match sensor_data_clone2.lock() {
            Ok(guard) => guard,
            Err(err) => {
                eprintln!("Recovering from poisoned sensor data mutex");
                err.into_inner()
            }
        };
        data.mag = Some(mag_msg);
    }) {
        return Err("Failed to subscribe to /mag".into());
    }

    println!("Run the following command in another terminal:");
    println!("GZ_SIM_RESOURCE_PATH=`pwd`/models gz sim -v 4 -r models/world.sdf");

    // This time tracking and mutexes everywhere are good enough for demo
    let mut last_time = std::time::Instant::now();
    loop {
        let now = std::time::Instant::now();
        let dt_actual = now.duration_since(last_time).as_secs_f64();

        if dt_actual >= dt {
            let mut ukf = ukf.lock().unwrap_or_else(|e| e.into_inner());

            let latest_gyro = {
                let data = sensor_data.lock().unwrap_or_else(|e| e.into_inner());
                data.imu.as_ref().map(|imu| {
                    Vector3::new(
                        imu.angular_velocity.x,
                        imu.angular_velocity.y,
                        imu.angular_velocity.z,
                    )
                })
            };

            if let Err(err) = ukf.predict(dt, latest_gyro.as_ref()) {
                eprintln!("UKF predict failed: {err:?}");
            }

            last_time = now;
        }

        {
            let ukf = match ukf.lock() {
                Ok(guard) => guard,
                Err(err) => {
                    eprintln!("Recovering from poisoned UKF mutex");
                    err.into_inner()
                }
            };
            let estimate = ukf.nominal_state();
            let estimated_attitude = attitude_component(estimate);
            let estimated_bias = bias_component(estimate);
            let estimated_euler = estimated_attitude.euler_angles();

            println!(
                "Est:  Roll={:6.2}° Pitch={:6.2}° Yaw={:6.2}°",
                estimated_euler.0.to_degrees(),
                estimated_euler.1.to_degrees(),
                estimated_euler.2.to_degrees()
            );
            println!(
                "... | Bias: {:.4} {:.4} {:.4}",
                estimated_bias.x, estimated_bias.y, estimated_bias.z
            );
        }
        tokio::time::sleep(Duration::from_secs_f64(dt)).await;
    }
}
