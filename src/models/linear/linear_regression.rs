use nalgebra::{DMatrix, DVector};

/// LinearRegression model implementation supporting both
/// normal equation and gradient descent methods
pub struct LinearRegression {
    /// Model coefficients (weights) including bias term
    coefficients: DVector<f64>,
    
    /// Optimization method for training
    /// Normal equation is preferred for smaller datasets (n_features < 10000)
    /// Gradient descent is better for larger datasets
    method: OptimizationMethod,
    
    /// Learning rate for gradient descent
    /// - Higher values: Faster learning but may overshoot
    /// - Lower values: More stable but slower convergence
    /// Recommended range: 0.001 to 0.1
    learning_rate: f64,
    
    /// Maximum number of iterations for gradient descent
    /// Increase if model hasn't converged, decrease if training is too slow
    /// Recommended range: 100 to 10000
    max_iterations: usize,
    
    /// Convergence criterion - training stops if cost change is below this value
    /// Recommended range: 1e-6 to 1e-4
    tolerance: f64,
    
    /// L2 regularization parameter (ridge regression when > 0)
    /// - Higher values: Stronger regularization, simpler model
    /// - Lower values: Weaker regularization, more complex model
    /// Recommended range: 0.01 to 10.0
    lambda: f64,
}

/// Available optimization methods for training
#[derive(Clone, Copy)]
pub enum OptimizationMethod {
    /// Analytical solution using normal equations
    /// Preferred for smaller datasets
    NormalEquation,
    
    /// Iterative solution using gradient descent
    /// Better for larger datasets
    GradientDescent,
}

impl LinearRegression {
    /// Create a new LinearRegression model with specified parameters
    pub fn new(
        method: OptimizationMethod,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
        lambda: f64,
    ) -> Self {
        LinearRegression {
            coefficients: DVector::zeros(0), // Will be initialized during fitting
            method,
            learning_rate,
            max_iterations,
            tolerance,
            lambda,
        }
    }

    /// Create a new LinearRegression model with default parameters
    pub fn default(method: OptimizationMethod) -> Self {
        Self::new(
            method,
            0.01,    // default learning rate
            1000,    // default max iterations
            1e-6,    // default tolerance
            0.0,     // default lambda (no regularization)
        )
    }

    /// Add bias term to feature matrix
    fn add_bias_term(x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut x_with_bias = DMatrix::from_element(x.nrows(), x.ncols() + 1, 1.0);
        x_with_bias.view_mut((0, 1), (x.nrows(), x.ncols())).copy_from(x);
        x_with_bias
    }

    /// Fit the model using the normal equation method
    fn fit_normal_equation(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<(), &'static str> {
        let n_features = x.ncols();

        // Add bias term to X
        let x_with_bias = Self::add_bias_term(x);

        // Normal equation with regularization: θ = (X^T X + λI)^(-1) X^T y
        let xtx = &x_with_bias.transpose() * &x_with_bias;
        
        // Add regularization term (don't regularize bias)
        let mut reg_matrix = DMatrix::identity(n_features + 1, n_features + 1);
        reg_matrix[(0, 0)] = 0.0;  // Don't regularize bias term
        reg_matrix *= self.lambda;

        match (&xtx + reg_matrix).try_inverse() {
            Some(xtx_inv) => {
                self.coefficients = &xtx_inv * &x_with_bias.transpose() * y;
                Ok(())
            },
            None => Err("Matrix is singular, cannot compute inverse. Try using gradient descent instead."),
        }
    }

    /// Fit the model using gradient descent
    fn fit_gradient_descent(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<Vec<f64>, &'static str> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Add bias term
        let x_with_bias = Self::add_bias_term(x);

        // Initialize coefficients if not already done
        if self.coefficients.len() != n_features + 1 {
            self.coefficients = DVector::zeros(n_features + 1);
        }

        let mut prev_cost = f64::INFINITY;
        let mut cost_history = Vec::with_capacity(self.max_iterations);

        for iteration in 0..self.max_iterations {
            // Compute predictions
            let predictions = &x_with_bias * &self.coefficients;
            
            // Compute error
            let errors = &predictions - y;
            
            // Compute gradients with regularization
            let mut gradients = &x_with_bias.transpose() * &errors / n_samples as f64;
            
            // Add L2 regularization gradient (exclude bias term)
            let mut reg_coefficients = self.coefficients.clone();
            reg_coefficients[0] = 0.0;  // Don't regularize bias
            gradients = gradients + self.lambda * &reg_coefficients;

            // Update coefficients
            self.coefficients = &self.coefficients - self.learning_rate * &gradients;

            // Compute cost (MSE + regularization)
            let mut cost = errors.dot(&errors) / (2.0 * n_samples as f64);
            
            // Add regularization term
            cost += self.lambda / 2.0 * self.coefficients
                .view((1, 0), (self.coefficients.len() - 1, 1))
                .norm_squared();

            cost_history.push(cost);

            // Check convergence
            if (prev_cost - cost).abs() < self.tolerance {
                println!("Converged at iteration {}", iteration);
                break;
            }
            prev_cost = cost;
        }

        Ok(cost_history)
    }

    /// Train the model on provided data
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<Option<Vec<f64>>, &'static str> {
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match");
        }

        match self.method {
            OptimizationMethod::NormalEquation => {
                self.fit_normal_equation(x, y)?;
                Ok(None)  // Normal equation doesn't produce cost history
            },
            OptimizationMethod::GradientDescent => {
                Ok(Some(self.fit_gradient_descent(x, y)?))
            },
        }
    }

    /// Make predictions for new data
    pub fn predict(&self, x: &DMatrix<f64>) -> Result<DVector<f64>, &'static str> {
        if x.ncols() != self.coefficients.len() - 1 {
            return Err("Number of features in X must match training data");
        }

        // Add bias term
        let x_with_bias = Self::add_bias_term(x);
        Ok(&x_with_bias * &self.coefficients)
    }

    /// Calculate R-squared score
    pub fn score(&self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<f64, &'static str> {
        let predictions = self.predict(x)?;
        let y_mean = y.mean();
        
        let total_sum_squares: f64 = y.iter()
            .map(|&yi| (yi - y_mean).powi(2))
            .sum();
            
        let residual_sum_squares: f64 = y.iter()
            .zip(predictions.iter())
            .map(|(&yi, &yhat_i)| (yi - yhat_i).powi(2))
            .sum();

        Ok(1.0 - (residual_sum_squares / total_sum_squares))
    }

    /// Get the model's coefficients
    pub fn get_coefficients(&self) -> &DVector<f64> {
        &self.coefficients
    }
}