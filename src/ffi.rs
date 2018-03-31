/*
Copyright 2017-2018 Peter Jin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

#[cfg(feature = "gpu")]
pub mod routines_gpu {

use gpudevicemem::ffi::routines_gpu::{KernelConfig};

use cuda::ffi::runtime::{CUstream_st};

/*#[derive(Clone, Copy)]
#[repr(C)]
pub struct KernelConfig {
  pub block_sz:     u32,
  pub max_block_ct: u32,
}*/

include!(concat!(env!("OUT_DIR"), "/routines_gpu_bind.rs"));

}
