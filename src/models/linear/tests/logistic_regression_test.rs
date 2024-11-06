use crate::models::linear::LogisticRegression;
use nalgebra::{DMatrix, DVector};

#[test]
fn test_binary_classification() {
    // Create sample data
    let x = DMatrix::from_row_slice(6, 2, &[
        1.0, 1.0,   // Class 1
        2.0, 2.0,   // Class 1
        2.0, 1.0,   // Class 1
        -1.0, -1.0, // Class 0
        -2.0, -2.0, // Class 0
        -1.0, -2.0, // Class 0
    ]);
    
    let y = DVector::from_row_slice(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

    let mut model = LogisticRegression::new(
        2,      // features
        0.1,    // learning rate
        1000,   // max iterations
        1e-6,   // tolerance
        None,   // no regularization
    );

    // Test fitting
    let cost_history = model.fit(&x, &y).unwrap();
    
    // Check if cost decreases
    assert!(cost_history.first().unwrap() > cost_history.last().unwrap(), 
            "Cost should decrease during training");

    // Test predictions on training data
    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), y.len(), "Should predict for all samples");
    
    // Check if predictions match training labels
    for (pred, actual) in predictions.iter().zip(y.iter()) {
        assert_eq!(pred.round(), *actual, "Prediction should match training label");
    }
}

#[test]
fn test_prediction_probabilities() {
    let x = DMatrix::from_row_slice(4, 2, &[
        2.0, 2.0,    // Strong class 1
        -2.0, -2.0,  // Strong class 0
        0.1, 0.1,    // Weak class 1
        -0.1, -0.1,  // Weak class 0
    ]);
    
    let y = DVector::from_row_slice(&[1.0, 0.0, 1.0, 0.0]);

    let mut model = LogisticRegression::new(2, 0.1, 1000, 1e-6, None);
    model.fit(&x, &y).unwrap();

    let probabilities = model.predict_proba(&x);
    
    // Check probability bounds
    for prob in probabilities.iter() {
        assert!(*prob >= 0.0 && *prob <= 1.0, 
                "Probabilities should be between 0 and 1");
    }

    // Strong predictions should be close to 0 or 1
    assert!(probabilities[0] > 0.9, "Strong class 1 should have high probability");
    assert!(probabilities[1] < 0.1, "Strong class 0 should have low probability");
}

#[test]
fn test_regularization() {
    let x = DMatrix::from_row_slice(6, 2, &[
        1.0, 1.0, 2.0, 2.0, 2.0, 1.0,
        -1.0, -1.0, -2.0, -2.0, -1.0, -2.0,
    ].chunks(2).collect::<Vec<_>>().concat());
    
    let y = DVector::from_row_slice(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

    // Train two models - one with regularization and one without
    let mut reg_model = LogisticRegression::new(2, 0.1, 1000, 1e-6, Some(1.0));
    let mut noreg_model = LogisticRegression::new(2, 0.1, 1000, 1e-6, None);

    reg_model.fit(&x, &y).unwrap();
    noreg_model.fit(&x, &y).unwrap();

    // Regularized weights should have smaller magnitude
    let reg_weights = reg_model.get_weights();
    let noreg_weights = noreg_model.get_weights();

    assert!(reg_weights.norm() < noreg_weights.norm(), 
            "Regularized weights should have smaller magnitude");
}

#[test]
#[should_panic(expected = "Number of samples in X and y must match")]
fn test_mismatched_dimensions() {
    let x = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let y = DVector::from_row_slice(&[1.0, 0.0]); // Wrong length

    let mut model = LogisticRegression::new(2, 0.1, 1000, 1e-6, None);
    model.fit(&x, &y).unwrap();
}