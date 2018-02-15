#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

#[cfg(feature = "gpu")]
pub mod routines_gpu {

use cuda::ffi::runtime::{CUstream_st};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct KernelConfig {
  pub block_sz:     u32,
  pub max_block_ct: u32,
}

include!(concat!(env!("OUT_DIR"), "/routines_gpu_bind.rs"));

}
