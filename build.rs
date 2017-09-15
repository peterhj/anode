extern crate bindgen;
extern crate gcc;
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

  gcc::Build::new()
    .cuda(true)
    .opt_level(2)
    .pic(true)
    .flag("-gencode").flag("arch=compute_37,code=sm_37")
    .flag("-gencode").flag("arch=compute_52,code=sm_52")
    .flag("-gencode").flag("arch=compute_52,code=compute_52")
    .flag("-prec-div=true")
    .flag("-prec-sqrt=true")
    .flag("-std=c++11")
    .flag("-Xcompiler").flag("-fno-strict-aliasing")
    .flag("-Xcompiler").flag("-Werror")
    .include("kernels_gpu")
    .include("/usr/local/cuda/include")
    .file("kernels_gpu/bcast_linear.cu")
    .file("kernels_gpu/flat_linear.cu")
    .file("kernels_gpu/flat_map.cu")
    .file("kernels_gpu/reduce.cu")
    .compile("libanode_kernels_gpu.a");

  bindgen::Builder::default()
    .header("kernels_gpu/lib.h")
    .link("anode_kernels_gpu")
    .whitelist_recursively(false)
    .whitelisted_function("anode_gpu_flat_mult_f32")
    .whitelisted_function("anode_gpu_flat_mult_add_f32")
    .whitelisted_function("anode_gpu_copy_flat_map_f32")
    .whitelisted_function("anode_gpu_modulus_flat_map_f32")
    .whitelisted_function("anode_gpu_square_flat_map_f32")
    .whitelisted_function("anode_gpu_positive_clip_flat_map_f32")
    .whitelisted_function("anode_gpu_unit_step_flat_map_f32")
    .whitelisted_function("anode_gpu_log_positive_clip_flat_map_f32")
    .whitelisted_function("anode_gpu_positive_reciprocal_flat_map_f32")
    .generate()
    .expect("bindgen failed to generate cuda kernel bindings")
    .write_to_file(out_dir.join("kernels_gpu_bind.rs"))
    .expect("bindgen failed to write cuda kernel bindings");
}
