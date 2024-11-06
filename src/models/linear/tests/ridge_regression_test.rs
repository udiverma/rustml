use super::super::RidgeRegression;
use nalgebra::{DMatrix, DVector};

#[test]
fn test_simple_ridge_regression() {
    // Create sample data
    let x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = DVector::from_row_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);  // y = 2x

    // Test with different regularization strengths
    let lambdas = [0.0, 0.1, 1.0, 10.0];
    
    for &lambda in &lambdas {
        let mut model = RidgeRegression::new(1, lambda);
        model.fit(&x, &y).unwrap();
        
        let coefficients = model.get_coefficients();
        
        // With λ=0, should be same as ordinary linear regression
        if lambda == 0.0 {
            assert!((coefficients[1] - 2.0).abs() < 1e-5, "Slope should be close to 2");
            assert!(coefficients[0].abs() < 1e-5, "Intercept should be close to 0");
        } else {
            // With regularization, coefficients should be smaller
            assert!(coefficients[1].abs() < 2.0, 
                "Regularized slope should be smaller than true slope");
        }
    }
}

#[test]
fn test_regularization_effect() {
    // Create data with noise
    let x = DMatrix::from_row_slice(10, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let y = DVector::from_row_slice(&[
        2.1, 4.2, 5.9, 8.1, 9.8, 12.2, 13.8, 16.1, 18.2, 20.3
    ]); // Noisy y ≈ 2x

    // Compare models with different regularization
    let mut unreg_model = RidgeRegression::new(1, 0.0);
    let mut reg_model = RidgeRegression::new(1, 1.0);

    unreg_model.fit(&x, &y).unwrap();
    reg_model.fit(&x, &y).unwrap();

    // Regularized model should have smaller coefficients
    let unreg_coef = unreg_model.get_coefficients();
    let reg_coef = reg_model.get_coefficients();
    
    assert!(reg_coef.norm() < unreg_coef.norm(), 
            "Regularized model should have smaller coefficients");
}

#[test]
fn test_predictions() {
    let train_x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let train_y = DVector::from_row_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut model = RidgeRegression::default(1);
    model.fit(&train_x, &train_y).unwrap();

    // Test predictions
    let test_x = DMatrix::from_row_slice(3, 1, &[6.0, 7.0, 8.0]);
    let predictions = model.predict(&test_x).unwrap();
    
    // Predictions should be close to 2x but slightly shrunk
    for (i, &pred) in predictions.iter().enumerate() {
        let x_val = (i as f64) + 6.0;
        let expected = 2.0 * x_val;
        assert!((pred - expected).abs() < 1.0, 
                "Prediction should be reasonably close to linear trend");
    }
}

#[test]
#[should_panic(expected = "Number of samples in X and y must match")]
fn test_mismatched_dimensions() {
    let x = DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]);
    let y = DVector::from_row_slice(&[2.0, 4.0]); // Wrong length

    let mut model = RidgeRegression::default(1);
    model.fit(&x, &y).unwrap();
}

#[test]
fn test_r_squared() {
    let x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = DVector::from_row_slice(&[2.1, 3.8, 6.2, 8.1, 9.8]); // Slightly noisy y ≈ 2x

    let mut model = RidgeRegression::new(1, 0.1);
    model.fit(&x, &y).unwrap();

    let r2 = model.score(&x, &y).unwrap();
    assert!(r2 > 0.95, "R² should be high for good fit");
    assert!(r2 <= 1.0, "R² should never exceed 1");
}