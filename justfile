# rec - quick speech to text.
# Install the runner with `cargo install just`.

# List the recipes.
default:
    @just --list

# Build with the local MLX model.
build:
    cargo build --release --features=local

# Install `rec` into ~/.cargo/bin, with the local MLX model.
install:
    cargo install --features=local --path . --force

# Install without the MLX model, API backends only.
install-api:
    cargo install --path . --force

# Lint.
check:
    cargo clippy --features=local --all-targets

# Run the tests.
test:
    cargo test --features=local

# Remove the build products.
clean:
    cargo clean
