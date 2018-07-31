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

use ::{Txn};

use arrayidx::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform, Normal};

use std::cell::{RefCell};
use std::rc::{Rc};

pub trait NormalLinearInit<T, R: Rng> {
  type RValue;

  fn normal_linear_init(std_dev: T, src: usize, dst: usize, rng: Rc<RefCell<R>>) -> Self::RValue;
}

impl<R: Rng + 'static> NormalLinearInit<f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn normal_linear_init(std_dev: f32, src_ch: usize, dst_ch: usize, rng: Rc<RefCell<R>>) -> Self::RValue {
    Rc::new(move |_, conn: GPUDeviceConn| {
      let shape = [dst_ch, src_ch];
      let mut h_arr = MemArray2d::<f32>::zeros(shape);
      {
        let dist = Normal::new(0.0, std_dev as f64);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        let mut rng = rng.borrow_mut();
        for x in xs.iter_mut() {
          *x = rng.sample(dist) as f32;
        }
      }
      let mut arr = GPUDeviceArray2d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

pub trait XavierLinearInit<T, R: Rng> {
  type RValue;

  fn xavier_linear_init(src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> XavierLinearInit<f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn xavier_linear_init(src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let shape = [dst_ch, src_ch];
      let mut h_arr = MemArray2d::<f32>::zeros(shape);
      {
        let half_width = (6.0 / (src_ch + dst_ch) as f64).sqrt();
        let dist = Uniform::new_inclusive(-half_width, half_width);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray2d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

pub trait KaimingConv2dInit<T, R: Rng> {
  type RValue;

  fn kaiming_conv2d_init(ker_sz: [usize; 2], src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> KaimingConv2dInit<f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn kaiming_conv2d_init(ker_sz: [usize; 2], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      let shape = [ker_sz[0], ker_sz[1], src_ch, dst_ch];
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(shape);
      {
        let mean = 0.0;
        let std = (2.0 / (ker_sz[0] * ker_sz[1] * src_ch) as f64).sqrt();
        let dist = Normal::new(mean, std);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray4d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

pub trait KaimingConv3dInit<T, R: Rng> {
  type RValue;

  fn kaiming_conv3d_init(ker_sz: [usize; 3], src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> KaimingConv3dInit<f32, R> for GPUDeviceArray5d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn kaiming_conv3d_init(ker_sz: [usize; 3], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      let shape = [ker_sz[0], ker_sz[1], ker_sz[2], src_ch, dst_ch];
      // TODO: seed the local rng here.
      let mut h_arr = MemArray5d::<f32>::zeros(shape);
      {
        let mean = 0.0;
        let std = (2.0 / (ker_sz[0] * ker_sz[1] * ker_sz[2] * src_ch) as f64).sqrt();
        let dist = Normal::new(mean, std);
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        for x in xs.iter_mut() {
          *x = dist.sample(&mut thread_rng()) as f32;
        }
      }
      let mut arr = GPUDeviceArray5d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}
