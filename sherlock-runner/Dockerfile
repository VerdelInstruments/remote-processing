# Sherlock Lambda — Rust pipeline in a container image
# Build in standard AL2023, then copy into Lambda runtime image

FROM --platform=linux/arm64 amazonlinux:2023 AS builder

RUN dnf install -y \
    gcc gcc-c++ make cmake git tar gzip \
    m4 autoconf automake libtool \
    zlib-devel \
    && dnf clean all

# Build HDF5 and NetCDF from source (not in AL2023 repos)
WORKDIR /deps

# HDF5 1.12.3 (hdf5-sys 0.8.1 doesn't support 1.14+)
RUN curl -fsSL -L https://github.com/HDFGroup/hdf5/releases/download/hdf5-1_12_3/hdf5-1_12_3.tar.gz | tar xz \
    && cd hdfsrc \
    && ./configure --prefix=/usr/local --enable-shared --disable-tests --disable-tools \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf hdfsrc

# NetCDF-C 4.9
RUN curl -fsSL -L https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.9.2.tar.gz | tar xz \
    && cd netcdf-c-4.9.2 \
    && CPPFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib" \
       ./configure --prefix=/usr/local --enable-shared --disable-dap --disable-byterange --disable-libxml2 \
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

ENV HDF5_DIR=/usr/local
ENV NETCDF_DIR=/usr/local
RUN cargo build --release

# --- Runtime image ---
FROM public.ecr.aws/lambda/provided:al2023-arm64

# Copy shared libs
COPY --from=builder /usr/local/lib/libnetcdf.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libhdf5.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libhdf5_hl.so* /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib

# Python 3 + sklearn for the RANSAC bridge (optional algorithm variant)
RUN dnf install -y python3 python3-pip && dnf clean all \
    && pip3 install --no-cache-dir numpy scikit-learn scipy xarray netCDF4

# Copy Python bridge scripts
COPY ransac_bridge.py /opt/ransac_bridge.py
COPY binning_bridge.py /opt/binning_bridge.py

# Copy binary
COPY --from=builder /build/target/release/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap

CMD ["bootstrap"]
