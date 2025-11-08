use estima::manifold::{
    ManifoldMeasurement, ManifoldProcess, composite::CompositeManifold,
    euclidean::EuclideanManifold, quaternion::UnitQuaternionManifold,
};
use nalgebra::{U3, U6, UnitQuaternion, Vector3, Vector6};

pub type AttitudeManifold = UnitQuaternionManifold<f64>;
pub type BiasManifold = EuclideanManifold<f64, U3>;
pub type AHRSState = CompositeManifold<f64, AttitudeManifold, BiasManifold, U3, U3>;

pub fn ahrs_state_new(attitude: UnitQuaternion<f64>, bias: Vector3<f64>) -> AHRSState {
    CompositeManifold::new(
        UnitQuaternionManifold::new(attitude),
        EuclideanManifold::new(bias),
    )
}

pub fn attitude_component(state: &AHRSState) -> &UnitQuaternion<f64> {
    state.first.as_quaternion()
}

pub fn bias_component(state: &AHRSState) -> &Vector3<f64> {
    state.second.as_vector()
}

#[derive(Clone)]
pub struct GyroscopeProcess;

impl ManifoldProcess<AHRSState, U3, f64> for GyroscopeProcess {
    fn predict(&self, state: &AHRSState, dt: f64, control: Option<&Vector3<f64>>) -> AHRSState {
        let bias = bias_component(state);
        let attitude = attitude_component(state);

        if let Some(gyro) = control {
            let corrected = gyro - bias;
            let delta = UnitQuaternion::from_scaled_axis(corrected * dt);
            ahrs_state_new(attitude * delta, *bias)
        } else {
            state.clone()
        }
    }
}

#[derive(Clone)]
pub struct AccelMagMeasurement {
    gravity_ref: Vector3<f64>,
    magnetic_ref: Vector3<f64>,
}

impl AccelMagMeasurement {
    /// For the demo purposes gravity and magnetic references are hardcoded
    /// with same values we use in Gazebo simulation
    pub fn new() -> Self {
        Self {
            gravity_ref: Vector3::new(0.0, 0.0, -9.81),
            magnetic_ref: Vector3::new(0.469, 0.121, 0.874).normalize(),
        }
    }

    /// In practice: if accel is near zero (freefall), measurement should be rejected, not faked.
    fn normalize_or_zero(vector: Vector3<f64>) -> Vector3<f64> {
        vector.try_normalize(1e-12).unwrap_or_else(Vector3::zeros)
    }
}

impl Default for AccelMagMeasurement {
    fn default() -> Self {
        Self::new()
    }
}

impl ManifoldMeasurement<AHRSState, U6, U6, f64> for AccelMagMeasurement {
    fn measure(&self, state: &AHRSState) -> Vector6<f64> {
        let attitude = attitude_component(state);
        let accel = -attitude.inverse().transform_vector(&self.gravity_ref);
        let mag = attitude.inverse().transform_vector(&self.magnetic_ref);

        let accel_norm = Self::normalize_or_zero(accel);
        let mag_norm = Self::normalize_or_zero(mag);

        Vector6::new(
            accel_norm.x,
            accel_norm.y,
            accel_norm.z,
            mag_norm.x,
            mag_norm.y,
            mag_norm.z,
        )
    }

    fn residual(&self, predicted: &Vector6<f64>, measured: &Vector6<f64>) -> Vector6<f64> {
        let pred_accel =
            Self::normalize_or_zero(Vector3::new(predicted[0], predicted[1], predicted[2]));
        let pred_mag =
            Self::normalize_or_zero(Vector3::new(predicted[3], predicted[4], predicted[5]));

        let meas_accel =
            Self::normalize_or_zero(Vector3::new(measured[0], measured[1], measured[2]));
        let meas_mag = Self::normalize_or_zero(Vector3::new(measured[3], measured[4], measured[5]));

        let accel_residual = pred_accel.cross(&meas_accel);
        let mag_residual = pred_mag.cross(&meas_mag);

        Vector6::new(
            accel_residual.x,
            accel_residual.y,
            accel_residual.z,
            mag_residual.x,
            mag_residual.y,
            mag_residual.z,
        )
    }

    fn innovation(&self, measured: &Vector6<f64>, predicted_mean: &Vector6<f64>) -> Vector6<f64> {
        self.residual(predicted_mean, measured)
    }
}
