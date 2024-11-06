mod linear_regression;
mod logistic_regression;
mod ridge_regression;

pub use linear_regression::{LinearRegression, OptimizationMethod};
pub use logistic_regression::LogisticRegression;
pub use ridge_regression::RidgeRegression;

#[cfg(test)]
mod tests;