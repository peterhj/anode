extern crate bindgen;
extern crate cc;
extern crate walkdir;

use walkdir::{WalkDir};

use std::env;
use std::path::{PathBuf};

fn main() {
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

  println!("cargo:rustc-link-search=native={}", out_dir.display());
  println!("cargo:rerun-if-changed=build.rs");

  let mut kernels_gpu_src_dir = PathBuf::from(manifest_dir.clone());
  kernels_gpu_src_dir.push("kernels_gpu");
  for entry in WalkDir::new(kernels_gpu_src_dir.to_str().unwrap()) {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }

  cc::Build::new()
    .cuda(true)
    .opt_level(2)
    .pic(true)
    .flag("-gencode").flag("arch=compute_37,code=sm_37")
    .flag("-gencode").flag("arch=compute_52,code=sm_52")
    .flag("-gencode").flag("arch=compute_60,code=sm_60")
    .flag("-gencode").flag("arch=compute_61,code=sm_61")
    .flag("-prec-div=true")
    .flag("-prec-sqrt=true")
    .flag("-std=c++11")
    .flag("-Xcompiler").flag("-fno-strict-aliasing")
    .flag("-Xcompiler").flag("-Werror")
    .include("kernels_gpu")
    .include("/usr/local/cuda/include")
    .file("kernels_gpu/bcast_flat_linear.cu")
    .file("kernels_gpu/flat_linear.cu")
    .file("kernels_gpu/flat_map.cu")
    .file("kernels_gpu/reduce.cu")
    .compile("libanode_kernels_gpu.a");

  bindgen::Builder::default()
    .header("kernels_gpu/lib.h")
    .link("anode_kernels_gpu")
    .whitelist_recursively(false)
    // "bcast_flat_linear.cu"
    .whitelisted_function("anode_gpu_bcast_flat_mult_I1b_I2ab_Oab_packed_f32")
    .whitelisted_function("anode_gpu_bcast_flat_mult_I1b_I2ab_I3b_Oab_packed_f32")
    .whitelisted_function("anode_gpu_bcast_flat_mult_I1b_I2abc_Oabc_packed_f32")
    .whitelisted_function("anode_gpu_bcast_flat_mult_I1b_I2abc_I3b_Oabc_packed_f32")
    // "flat_linear.cu"
    .whitelisted_function("anode_gpu_flat_mult_f32")
    .whitelisted_function("anode_gpu_flat_mult_add_f32")
    // "flat_map.cu"
    .whitelisted_function("anode_gpu_copy_flat_map_f32")
    .whitelisted_function("anode_gpu_modulus_flat_map_f32")
    .whitelisted_function("anode_gpu_square_flat_map_f32")
    .whitelisted_function("anode_gpu_positive_clip_flat_map_f32")
    .whitelisted_function("anode_gpu_unit_step_flat_map_f32")
    .whitelisted_function("anode_gpu_log_positive_clip_flat_map_f32")
    .whitelisted_function("anode_gpu_positive_reciprocal_flat_map_f32")
    // "reduce.cu"
    .whitelisted_function("anode_gpu_sum_reduce_I1ab_Ob_packed_deterministic_f32")
    .generate()
    .expect("bindgen failed to generate cuda kernel bindings")
    .write_to_file(out_dir.join("kernels_gpu_bind.rs"))
    .expect("bindgen failed to write cuda kernel bindings");
}
