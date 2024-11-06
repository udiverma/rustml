use rustml::models::linear::{LinearRegression, OptimizationMethod};
use nalgebra::{DMatrix, DVector};

fn main() {
    println!("Linear Regression Example");
    println!("------------------------");

    // Create sample data (y = 2x + 1 with some noise)
    let x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = DVector::from_row_slice(&[2.8, 5.1, 7.2, 8.8, 11.3]);

    println!("Training Data:");
    println!("X values: {}", x);
    println!("Y values: {}", y);

    // Create and train model
    let mut model = LinearRegression::new(
        OptimizationMethod::NormalEquation,
        0.01,    // learning rate
        1000,    // max iterations
        1e-6,    // tolerance
        0.0,     // no regularization
    );

    // Fit the model
    match model.fit(&x, &y) {
        Ok(_) => println!("\nModel trained successfully!"),
        Err(e) => {
            println!("Error training model: {}", e);
            return;
        }
    }

    // Print model coefficients
    let coefficients = model.get_coefficients();
    println!("Model coefficients:");
    println!("Intercept: {:.4}", coefficients[0]);
    println!("Slope: {:.4}", coefficients[1]);

    // Make predictions
    let test_x = DMatrix::from_row_slice(3, 1, &[6.0, 7.0, 8.0]);
    match model.predict(&test_x) {
        Ok(predictions) => {
            println!("\nPredictions for new data:");
            println!("X: {}", test_x);
            println!("Predicted y: {}", predictions);
        },
        Err(e) => println!("Error making predictions: {}", e),
    }

    // Calculate R-squared score
    match model.score(&x, &y) {
        Ok(r2_score) => println!("\nR-squared score: {:.4}", r2_score),
        Err(e) => println!("Error calculating R-squared: {}", e),
    }
}