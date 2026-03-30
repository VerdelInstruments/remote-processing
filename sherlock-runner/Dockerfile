# Sherlock Lambda — Rust pipeline in a container image
# Build in standard AL2023, then copy into Lambda runtime image

FROM amazonlinux:2023 AS builder

# Enable EPEL for netcdf/hdf5 packages
RUN dnf install -y \
    gcc gcc-c++ make cmake git curl tar gzip \
    zlib-devel \
    && dnf clean all

# Build HDF5 and NetCDF from source (not in AL2023 repos)
WORKDIR /deps

# HDF5 1.14
RUN curl -fsSL https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.5/hdf5-1.14.5.tar.gz | tar xz \
    && cd hdf5-1.14.5 \
    && ./configure --prefix=/usr/local --enable-shared --disable-tests --disable-tools \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf hdf5-1.14.5

# NetCDF-C 4.9
RUN curl -fsSL https://github.com/Unidata/netcdf-c/releases/download/v4.9.2/netcdf-c-4.9.2.tar.gz | tar xz \
    && cd netcdf-c-4.9.2 \
    && CPPFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib" \
       ./configure --prefix=/usr/local --enable-shared --disable-dap --disable-byterange \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf netcdf-c-4.9.2

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Build the Lambda binary
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

RUN cargo build --release

# --- Runtime image ---
FROM public.ecr.aws/lambda/provided:al2023-arm64

# Copy shared libs
COPY --from=builder /usr/local/lib/libnetcdf.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libhdf5.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libhdf5_hl.so* /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy binary
COPY --from=builder /build/target/release/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap

CMD ["bootstrap"]
