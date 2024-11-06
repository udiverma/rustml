# rustml

A pure Rust implementation of fundamental machine learning algorithms. This library provides fast, reliable, and type-safe implementations of common ML algorithms with a focus on clarity and usability.

## Currently Implemented Algorithms

### Linear Models
- **Linear Regression**
  - Normal Equation method
  - Gradient Descent optimization
  - L2 regularization (Ridge)
  - R-squared scoring

- **Logistic Regression**
  - Binary classification
  - L2 regularization
  - Probability estimates
  - Gradient Descent optimization

## Project Structure
```
rustml/
├── src/
│   ├── models/
│   │   ├── linear/
│   │   │   ├── linear_regression.rs     # Linear Regression implementation
│   │   │   ├── logistic_regression.rs   # Logistic Regression implementation
│   │   │   ├── mod.rs                   # Module exports
│   │   │   └── tests/                   # Unit tests
│   │   │       ├── linear_regression_test.rs
│   │   │       └── logistic_regression_test.rs
│   │   ├── classification/              # Future classification algorithms
│   │   ├── clustering/                  # Future clustering algorithms
│   │   ├── dimensionality_reduction/    # Future dimensionality reduction
│   │   └── mod.rs
│   └── lib.rs
├── examples/                            # Usage examples
│   ├── linear_regression.rs
│   └── logistic_regression.rs
└── Cargo.toml
```

## Dependencies
- `nalgebra`: Linear algebra operations
- `rand`: Random number generation
- `thiserror`: Error handling
- `serde`: (Optional) Serialization support

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
rustml = { git = "https://github.com/yourusername/rustml" }
```

## Usage Examples

### Linear Regression
```rust
use rustml::models::linear::{LinearRegression, OptimizationMethod};
use nalgebra::{DMatrix, DVector};

fn main() {
    // Create sample data (y = 2x + 1 with noise)
    let x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = DVector::from_row_slice(&[2.8, 5.1, 7.2, 8.8, 11.3]);

    // Create and train model
    let mut model = LinearRegression::new(
        OptimizationMethod::NormalEquation,
        0.01,    // learning rate
        1000,    // max iterations
        1e-6,    // tolerance
        0.0,     // no regularization
    );

    // Fit the model
    model.fit(&x, &y).expect("Failed to train model");

    // Make predictions
    let predictions = model.predict(&x).expect("Failed to make predictions");
    println!("Predictions: {}", predictions);

    // Calculate R-squared score
    let r2 = model.score(&x, &y).expect("Failed to calculate R²");
    println!("R² Score: {:.4}", r2);
}
```

### Logistic Regression
```rust
use rustml::models::linear::LogisticRegression;
use nalgebra::{DMatrix, DVector};

fn main() {
    // Create binary classification data
    let x = DMatrix::from_row_slice(4, 2, &[
        1.0, 1.0,   // Class 1
        2.0, 2.0,   // Class 1
        -1.0, -1.0, // Class 0
        -2.0, -2.0, // Class 0
    ]);
    let y = DVector::from_row_slice(&[1.0, 1.0, 0.0, 0.0]);

    // Create and train model
    let mut model = LogisticRegression::new(
        2,          // features
        0.1,        // learning rate
        1000,       // max iterations
        1e-6,       // tolerance
        Some(0.01), // L2 regularization
    );

    model.fit(&x, &y).expect("Failed to train model");

    // Get probabilities and class predictions
    let probabilities = model.predict_proba(&x);
    let predictions = model.predict(&x);
}
```

## Running Examples
```bash
# Run linear regression example
cargo run --example linear_regression

# Run logistic regression example
cargo run --example logistic_regression
```

## Running Tests
```bash
# Run all tests
cargo test

# Run specific model tests
cargo test linear_regression
cargo test logistic_regression

# Run tests with output
cargo test -- --nocapture

# Run tests with more details
cargo test -- --show-output
```

## Development Commands
```bash
# Check compilation
cargo check

# Build library
cargo build

# Build with optimizations
cargo build --release

# Run specific example with optimizations
cargo run --release --example linear_regression
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
MIT License - See LICENSE file for details

## Contact
Udit Verma - [@udiverma](https://www.linkedin.com/in/udiverma)

Project Link: [https://github.com/udiverma/rustml](https://github.com/udiverma/rustml)