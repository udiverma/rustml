use rustml::models::linear::RidgeRegression;
use nalgebra::{DMatrix, DVector};
use rand::Rng;

fn generate_noisy_data(samples: usize, noise_level: f64) -> (DMatrix<f64>, DVector<f64>) {
    let mut rng = rand::thread_rng();
    
    // Create input features
    let x = DMatrix::from_fn(samples, 1, |i, _| (i as f64) * 0.5);
    
    // Generate target values: y = 2x + 1 + noise
    let y = DVector::from_fn(samples, |i, _| {
        let x_val = i as f64 * 0.5;
        let true_y = 2.0 * x_val + 1.0;
        true_y + rng.gen_range(-noise_level..noise_level)
    });

    (x, y)
}

fn main() {
    println!("Ridge Regression Example");
    println!("----------------------");

    // Generate training data with noise
    let (x_train, y_train) = generate_noisy_data(20, 1.0);
    
    println!("Training Data:");
    println!("X values: {}", x_train);
    println!("Y values: {}", y_train);

    // Try different regularization strengths
    let lambdas = [0.0, 0.1, 1.0, 10.0];

    for &lambda in &lambdas {
        println!("\nTraining with λ = {}", lambda);
        
        // Create and train model
        let mut model = RidgeRegression::new(1, lambda);
        match model.fit(&x_train, &y_train) {
            Ok(_) => {
                let coefficients = model.get_coefficients();
                println!("Model coefficients:");
                println!("  Intercept: {:.4}", coefficients[0]);
                println!("  Slope: {:.4}", coefficients[1]);

                // Calculate R-squared score
                if let Ok(r2_score) = model.score(&x_train, &y_train) {
                    println!("R-squared score: {:.4}", r2_score);
                }

                // Make predictions on test data
                let x_test = DMatrix::from_row_slice(5, 1, &[5.0, 6.0, 7.0, 8.0, 9.0]);
                if let Ok(predictions) = model.predict(&x_test) {
                    println!("\nPredictions for new data:");
                    println!("X: {}", x_test);
                    println!("Predicted y: {}", predictions);

                    // Calculate true values for comparison
                    let true_y: Vec<f64> = x_test.column(0)
                        .iter()
                        .map(|&x| 2.0 * x + 1.0)
                        .collect();
                    println!("True y (without noise): {:.4}", DVector::from_vec(true_y));
                }
            },
            Err(e) => println!("Error training model: {}", e),
        }

        println!("\nEffect of regularization:");
        println!("- λ = 0.0: No regularization (standard linear regression)");
        println!("- λ = 0.1: Weak regularization (slight coefficient shrinkage)");
        println!("- λ = 1.0: Moderate regularization");
        println!("- λ = 10.0: Strong regularization (higher bias, lower variance)");
    }

    // Demonstrate overfitting scenario
    println!("\nOverfitting Demonstration");
    println!("-----------------------");
    
    // Generate very noisy data
    let (x_noisy, y_noisy) = generate_noisy_data(20, 2.0);
    
    // Compare unregularized vs regularized
    let mut unreg_model = RidgeRegression::new(1, 0.0);
    let mut reg_model = RidgeRegression::new(1, 1.0);

    unreg_model.fit(&x_noisy, &y_noisy).unwrap();
    reg_model.fit(&x_noisy, &y_noisy).unwrap();

    println!("\nWith noisy data:");
    println!("Unregularized coefficients: {:.4}", unreg_model.get_coefficients());
    println!("Regularized coefficients: {:.4}", reg_model.get_coefficients());
    
    println!("\nUnregularized R²: {:.4}", unreg_model.score(&x_noisy, &y_noisy).unwrap());
    println!("Regularized R²: {:.4}", reg_model.score(&x_noisy, &y_noisy).unwrap());

    // Generate test data without noise
    let x_test = DMatrix::from_row_slice(5, 1, &[10.0, 11.0, 12.0, 13.0, 14.0]);
    let y_test = DVector::from_fn(5, |i, _| {
        let x_val = 10.0 + i as f64;
        2.0 * x_val + 1.0  // True relationship without noise
    });

    println!("\nPerformance on clean test data:");
    println!("Unregularized R²: {:.4}", unreg_model.score(&x_test, &y_test).unwrap());
    println!("Regularized R²: {:.4}", reg_model.score(&x_test, &y_test).unwrap());
}