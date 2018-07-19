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

  {
    // TODO
    //println!("cargo:rustc-link-lib=static=anode_routines_omp");
  }

  #[cfg(feature = "gpu")]
  {
    println!("cargo:rustc-link-lib=static=anode_routines_gpu");

    let mut routines_gpu_src_dir = PathBuf::from(manifest_dir.clone());
    routines_gpu_src_dir.push("routines_gpu");
    for entry in WalkDir::new(routines_gpu_src_dir.to_str().unwrap()) {
      let entry = entry.unwrap();
      println!("cargo:rerun-if-changed={}", entry.path().display());
    }

    cc::Build::new()
      .cuda(true)
      .opt_level(2)
      .pic(true)
      .flag("-gencode").flag("arch=compute_35,code=sm_35")
      .flag("-gencode").flag("arch=compute_37,code=sm_37")
      .flag("-gencode").flag("arch=compute_52,code=sm_52")
      .flag("-gencode").flag("arch=compute_60,code=sm_60")
      .flag("-gencode").flag("arch=compute_61,code=sm_61")
      .flag("-gencode").flag("arch=compute_70,code=sm_70")
      .flag("-prec-div=true")
      .flag("-prec-sqrt=true")
      .flag("-std=c++11")
      .flag("-Xcompiler").flag("-fno-strict-aliasing")
      .flag("-Xcompiler").flag("-Wall")
      .flag("-Xcompiler").flag("-Werror")
      .include("routines_gpu")
      .include("/usr/local/cuda/include")
      .file("routines_gpu/batch_norm.cu")
      .file("routines_gpu/batch_norm_external.cu")
      .file("routines_gpu/flat_join.cu")
      .file("routines_gpu/flat_map.cu")
      .file("routines_gpu/quantize.cu")
      .file("routines_gpu/softmax.cu")
      .compile("libanode_routines_gpu.a");

    bindgen::Builder::default()
      .header("routines_gpu/lib.h")
      .whitelist_recursively(false)
      // "batch_norm.cu"
      .whitelist_function("anode_gpu_batch_mean_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_mean_bwd_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_mean_bwd_3d1_packed_accumulate_f32")
      .whitelist_function("anode_gpu_batch_var_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_var_bwd_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_var_bwd_3d1_packed_accumulate_f32")
      .whitelist_function("anode_gpu_batch_var_bwd_mean_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_var_bwd_mean_3d1_packed_accumulate_f32")
      .whitelist_function("anode_gpu_batch_norm_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_norm_bwd_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_norm_bwd_3d1_packed_accumulate_f32")
      .whitelist_function("anode_gpu_batch_norm_bwd_mean_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_norm_bwd_mean_3d1_packed_accumulate_f32")
      .whitelist_function("anode_gpu_batch_norm_bwd_var_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_norm_bwd_var_3d1_packed_accumulate_f32")
      .whitelist_function("anode_gpu_batch_norm_bwd_var_v2_3d1_packed_f32")
      .whitelist_function("anode_gpu_batch_norm_bwd_var_v2_3d1_packed_accumulate_f32")
      // "flat_map.cu"
      .whitelist_function("anode_gpu_copy_flat_map_f32")
      .whitelist_function("anode_gpu_modulus_flat_map_f32")
      .whitelist_function("anode_gpu_square_flat_map_f32")
      .whitelist_function("anode_gpu_positive_clip_flat_map_f32")
      .whitelist_function("anode_gpu_positive_clip_flat_map_bwd_f32")
      .whitelist_function("anode_gpu_unit_step_flat_map_f32")
      .whitelist_function("anode_gpu_log_positive_clip_flat_map_f32")
      .whitelist_function("anode_gpu_positive_reciprocal_flat_map_f32")
      .whitelist_function("anode_gpu_tanh_flat_map_f32")
      .whitelist_function("anode_gpu_rcosh2_flat_map_f32")
      .whitelist_function("anode_gpu_M1_copy_map_M2_unit_step_map_R_product_reduce_flat_join_f32")
      // "quantize.cu"
      .whitelist_function("anode_gpu_dequantize_u8_packed_f32")
      // "softmax.cu"
      .whitelist_function("anode_gpu_softmax_packed_block_f32")
      .whitelist_function("anode_gpu_softmax_packed_deterministic_f32")
      .whitelist_function("anode_gpu_softmax_cat_nll_packed_f32")
      .whitelist_function("anode_gpu_softmax_cat_nll_bwd_packed_f32")
      .whitelist_function("anode_gpu_softmax_cat_nll_bwd_packed_accumulate_f32")
      .generate()
      .expect("bindgen failed to generate cuda kernel bindings")
      .write_to_file(out_dir.join("routines_gpu_bind.rs"))
      .expect("bindgen failed to write cuda kernel bindings");
  }
}
