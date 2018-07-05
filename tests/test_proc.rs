extern crate anode;
//extern crate gpudevicemem;
extern crate memarray;
extern crate rand;

use anode::*;
use anode::context::*;
use anode::log::*;
use anode::ops::*;
//use anode::ops_gpu::*;
use anode::proc_thread::*;
use anode::utils::*;
//use gpudevicemem::*;
//use gpudevicemem::array::*;
use memarray::*;
use rand::*;

use std::rc::{Rc};
use std::thread::{sleep};
use std::time::{Duration};

#[test]
fn test_proc_thread_io() {
  println!();
  let mut pg = ThreadProcGroup::new(1);
  for p in pg {
    p.spawn(|proc| {
      let src_buf: Vec<i32> = vec![0, 0, 0];
      let mut dst_buf: Vec<i32> = vec![0, 0, 0];
      let (tx, rx) = proc.allreduce_sum();
      tx.send(&src_buf);
      rx.recv(&mut dst_buf);
    });
  }
}
