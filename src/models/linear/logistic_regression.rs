use nalgebra::{DMatrix, DVector};

/// LogisticRegression model for binary classification
/// 
/// # Hyperparameters
/// All hyperparameters can be tuned during model initialization
pub struct LogisticRegression {
    /// Model weights including bias term
    weights: DVector<f64>,
    
    /// Learning rate for gradient descent
    /// - Higher values: Faster learning but may overshoot
    /// - Lower values: More stable but slower convergence
    /// Recommended range: 0.001 to 0.1
    learning_rate: f64,
    
    /// Maximum number of training iterations
    /// Increase if model hasn't converged, decrease if training is too slow
    /// Recommended range: 100 to 10000
    max_iterations: usize,
    
    /// Convergence criterion - training stops if cost change is below this value
    /// - Higher values: Faster training but less precise
    /// - Lower values: More precise but slower training
    /// Recommended range: 1e-6 to 1e-4
    tolerance: f64,

    /// Optional L2 regularization parameter (None if no regularization)
    /// - Higher values: Stronger regularization, simpler model
    /// - Lower values: Weaker regularization, more complex model
    /// Recommended range: 0.01 to 10.0
    lambda: Option<f64>,
}

impl LogisticRegression {
    /// Create a new LogisticRegression model
    pub fn new(
        features: usize,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
        lambda: Option<f64>,
    ) -> Self {
        LogisticRegression {
            weights: DVector::zeros(features + 1), // +1 for bias term
            learning_rate,
            max_iterations,
            tolerance,
            lambda,
        }
    }

    /// Create a new LogisticRegression model with default parameters
    pub fn default(features: usize) -> Self {
        Self::new(
            features,
            0.01,    // default learning rate
            1000,    // default max iterations
            1e-6,    // default tolerance
            None,    // no regularization by default
        )
    }

    /// Sigmoid activation function
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Add bias term to feature matrix
    fn add_bias_term(x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut x_with_bias = DMatrix::from_element(x.nrows(), x.ncols() + 1, 1.0);
        x_with_bias.view_mut((0, 1), (x.nrows(), x.ncols())).copy_from(x);
        x_with_bias
    }

    /// Calculate log loss
    fn compute_log_loss(&self, h: &DVector<f64>, y: &DVector<f64>, n_samples: usize) -> f64 {
        let epsilon = 1e-15; // Small constant to prevent log(0)
        let h_clipped = h.map(|x| x.max(epsilon).min(1.0 - epsilon));
        
        let term1 = y.component_mul(&h_clipped.map(|x| x.ln()));
        let term2 = &(DVector::from_element(y.len(), 1.0) - y)
            .component_mul(&h_clipped.map(|x| (1.0 - x).ln()));
        
        -1.0 / n_samples as f64 * (term1 + term2).sum()
    }

    /// Train the model on provided data
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<Vec<f64>, &'static str> {
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match");
        }

        // Add bias term (column of ones) to X
        let x_with_bias = Self::add_bias_term(x);
        let n_samples = x.nrows();
        
        let mut prev_cost = f64::INFINITY;
        let mut cost_history = Vec::with_capacity(self.max_iterations);
        
        for iteration in 0..self.max_iterations {
            // Forward pass
            let z = &x_with_bias * &self.weights;
            let h: DVector<f64> = z.map(|x| Self::sigmoid(x));
            
            // Compute gradients
            let mut gradients = &x_with_bias.transpose() * (&h - y) / n_samples as f64;
            
            // Add L2 regularization if specified
            if let Some(lambda) = self.lambda {
                let mut reg_weights = self.weights.clone();
                reg_weights[0] = 0.0;  // Don't regularize bias term
                gradients = gradients + lambda * &reg_weights;
            }
            
            // Update weights using gradient descent
            self.weights = &self.weights - self.learning_rate * &gradients;
            
            // Compute cost (log loss with optional regularization)
            let mut cost = self.compute_log_loss(&h, y, n_samples);
            
            // Add L2 regularization term to cost if specified
            if let Some(lambda) = self.lambda {
                let reg_term = lambda / (2.0 * n_samples as f64) * 
                    self.weights.view((1, 0), (self.weights.len() - 1, 1)).norm_squared();
                cost += reg_term;
            }
            
            cost_history.push(cost);
            
            // Check for convergence
            if (prev_cost - cost).abs() < self.tolerance {
                println!("Converged at iteration {}", iteration);
                break;
            }
            prev_cost = cost;
        }
        
        Ok(cost_history)
    }

    /// Get probability predictions
    pub fn predict_proba(&self, x: &DMatrix<f64>) -> DVector<f64> {
        let x_with_bias = Self::add_bias_term(x);
        let z = &x_with_bias * &self.weights;
        z.map(|x| Self::sigmoid(x))
    }

    /// Get binary predictions using 0.5 as the default threshold
    pub fn predict(&self, x: &DMatrix<f64>) -> DVector<f64> {
        self.predict_proba(x)
            .map(|prob| if prob >= 0.5 { 1.0 } else { 0.0 })
    }

    /// Get the model's weights
    pub fn get_weights(&self) -> &DVector<f64> {
        &self.weights
    }
}