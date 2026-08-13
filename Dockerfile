FROM rust:1.89-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY defaults ./defaults
COPY assets ./assets
RUN cargo build --locked --release

FROM debian:bookworm-slim
# The Rust builder already contains the Mozilla CA bundle. Copy it instead of
# invoking apt in the runtime stage; this keeps production builds reliable on
# hosts with small Docker overlay/package-cache allocations.
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
RUN useradd --system --uid 10001 --create-home teleforge
WORKDIR /app
COPY --from=builder /build/target/release/teleforge /usr/local/bin/teleforge
COPY config.example.yaml /app/config.yaml
RUN mkdir -p /app/data && chown -R teleforge:teleforge /app
USER teleforge
EXPOSE 8080
ENTRYPOINT ["teleforge", "--config", "/app/config.yaml"]
