use nalgebra::{DMatrix, DVector};

/// Ridge Regression model implementation
/// Adds L2 regularization to linear regression: min(||y - Xw||² + λ||w||²)
pub struct RidgeRegression {
    /// Model coefficients (weights) including bias term
    coefficients: DVector<f64>,
    
    /// L2 regularization strength
    /// - Higher values: Stronger regularization, simpler model
    /// - Lower values: Weaker regularization, more complex model
    /// Recommended range: 0.01 to 10.0
    lambda: f64,
}

impl RidgeRegression {
    /// Create a new RidgeRegression model
    pub fn new(features: usize, lambda: f64) -> Self {
        RidgeRegression {
            coefficients: DVector::zeros(features + 1), // +1 for bias term
            lambda,
        }
    }

    /// Create a default RidgeRegression model with moderate regularization
    pub fn default(features: usize) -> Self {
        Self::new(features, 1.0)
    }

    /// Add bias term to feature matrix
    fn add_bias_term(x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut x_with_bias = DMatrix::from_element(x.nrows(), x.ncols() + 1, 1.0);
        x_with_bias.view_mut((0, 1), (x.nrows(), x.ncols())).copy_from(x);
        x_with_bias
    }

    /// Train the model using the closed-form solution
    /// w = (X^T X + λI)^(-1) X^T y
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<(), &'static str> {
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match");
        }

        let n_features = x.ncols();

        // Add bias term to X
        let x_with_bias = Self::add_bias_term(x);

        // Compute X^T X
        let xtx = &x_with_bias.transpose() * &x_with_bias;
        
        // Create regularization matrix (don't regularize bias term)
        let mut reg_matrix = DMatrix::identity(n_features + 1, n_features + 1);
        reg_matrix[(0, 0)] = 0.0;  // Don't regularize bias term
        reg_matrix *= self.lambda;

        // Compute (X^T X + λI)^(-1) X^T y
        match (&xtx + reg_matrix).try_inverse() {
            Some(xtx_inv) => {
                self.coefficients = &xtx_inv * &x_with_bias.transpose() * y;
                Ok(())
            },
            None => Err("Matrix is singular, cannot compute inverse"),
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