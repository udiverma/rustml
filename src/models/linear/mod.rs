mod linear_regression;
mod logistic_regression;

pub use linear_regression::{LinearRegression, OptimizationMethod};
pub use logistic_regression::LogisticRegression;

#[cfg(test)]
mod tests;