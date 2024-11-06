use rustml::models::linear::LogisticRegression;
use nalgebra::{DMatrix, DVector};

fn main() {
    println!("Logistic Regression Example");
    println!("--------------------------");

    // Create sample data for binary classification
    let x = DMatrix::from_row_slice(6, 2, &[
        1.0, 1.0,   // Class 1
        2.0, 2.0,   // Class 1
        2.0, 1.0,   // Class 1
        -1.0, -1.0, // Class 0
        -2.0, -2.0, // Class 0
        -1.0, -2.0, // Class 0
    ]);
    
    let y = DVector::from_row_slice(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

    println!("Training Data:");
    println!("X values:\n{}", x);
    println!("Y values: {}", y);

    // Create and train model
    let mut model = LogisticRegression::new(
        2,          // number of features
        0.1,        // learning rate
        1000,       // max iterations
        1e-6,       // tolerance
        Some(0.01), // L2 regularization
    );

    // Fit the model
    match model.fit(&x, &y) {
        Ok(cost_history) => {
            println!("\nModel trained successfully!");
            println!("Final cost: {:.6}", cost_history.last().unwrap());
        },
        Err(e) => {
            println!("Error training model: {}", e);
            return;
        }
    }

    // Make predictions on new data
    let test_x = DMatrix::from_row_slice(4, 2, &[
        3.0, 3.0,    // Should be Class 1
        2.0, 1.5,    // Should be Class 1
        -2.0, -1.5,  // Should be Class 0
        -3.0, -3.0,  // Should be Class 0
    ]);

    println!("\nPredictions for new data:");
    println!("Test X:\n{}", test_x);
    println!("\nProbabilities:");
    println!("{}", model.predict_proba(&test_x));
    println!("\nClassifications:");
    println!("{}", model.predict(&test_x));
}