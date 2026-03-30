# Sherlock Lambda — Rust pipeline in a container image
# Two-stage build: compile in AL2023, then copy binary + shared libs

FROM public.ecr.aws/lambda/provided:al2023-arm64 AS builder

# Build tools + NetCDF/HDF5 dev libs
RUN dnf install -y gcc gcc-c++ make cmake \
    netcdf-devel hdf5-devel zlib-devel \
    && dnf clean all

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

RUN cargo build --release

# --- Runtime image ---
FROM public.ecr.aws/lambda/provided:al2023-arm64

# Install NetCDF runtime libs (no -devel needed)
RUN dnf install -y netcdf hdf5 && dnf clean all

COPY --from=builder /build/target/release/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap

CMD ["bootstrap"]
