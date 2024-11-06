use crate::models::linear::{LinearRegression, OptimizationMethod};
use nalgebra::{DMatrix, DVector};

#[test]
fn test_simple_regression() {
    // Create sample data
    let x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = DVector::from_row_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);  // y = 2x

    let mut model = LinearRegression::default(OptimizationMethod::NormalEquation);
    
    // Test fitting
    match model.fit(&x, &y) {
        Ok(_) => {
            let coefficients = model.get_coefficients();
            assert!((coefficients[1] - 2.0).abs() < 1e-5, "Slope should be close to 2");
            assert!(coefficients[0].abs() < 1e-5, "Intercept should be close to 0");
        },
        Err(e) => panic!("Failed to fit model: {}", e)
    }
}

#[test]
fn test_predictions() {
    let train_x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let train_y = DVector::from_row_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = LinearRegression::default(OptimizationMethod::NormalEquation);
    model.fit(&train_x, &train_y).unwrap();

    // Test predictions
    let test_x = DMatrix::from_row_slice(3, 1, &[6.0, 7.0, 8.0]);
    let predictions = model.predict(&test_x).unwrap();
    
    // Expected values: [12.0, 14.0, 16.0]
    for (i, &pred) in predictions.iter().enumerate() {
        let expected = 2.0 * (i as f64 + 6.0);
        assert!((pred - expected).abs() < 1e-5, 
                "Prediction {} differs from expected {}", pred, expected);
    }
}

#[test]
fn test_r_squared() {
    let x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = DVector::from_row_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = LinearRegression::default(OptimizationMethod::NormalEquation);
    model.fit(&x, &y).unwrap();

    let r2 = model.score(&x, &y).unwrap();
    assert!((r2 - 1.0).abs() < 1e-5, "R² should be close to 1.0 for perfect fit");
}

#[test]
#[should_panic(expected = "Number of samples in X and y must match")]
fn test_mismatched_dimensions() {
    let x = DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]);
    let y = DVector::from_row_slice(&[2.0, 4.0]); // Wrong length

    let mut model = LinearRegression::default(OptimizationMethod::NormalEquation);
    model.fit(&x, &y).unwrap();
}