#![feature(alloc_system)]

extern crate anode;
extern crate colorimage;
extern crate gpudevicemem;
extern crate memarray;
extern crate rand;
extern crate rngtape;
extern crate sharedmem;
extern crate superdata;

use anode::*;
use anode::log::*;
use anode::ops::*;
use anode::proc_dist::*;
use anode::utils::*;
use colorimage::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform, Normal, StandardNormal};
use rngtape::*;
use sharedmem::*;
use superdata::*;
use superdata::datasets::mnist::*;
use superdata::image::*;
use superdata::utils::*;

use std::cell::{RefCell};
use std::cmp::{max, min};
use std::env;
use std::path::{PathBuf};
use std::rc::{Rc};
use std::sync::{Arc};

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
        let dist = StandardNormal;
        let mut v = h_arr.as_view_mut();
        let xs = v.flat_slice_mut().unwrap();
        let mut rng = rng.borrow_mut();
        for x in xs.iter_mut() {
          *x = rng.sample(dist) as f32 * std_dev;
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
        println!("DEBUG: kaiming conv2d: {} {:?} {:?}",
            xs.len(), &xs[ .. 10], &xs[xs.len() - 10 .. ]);
      }
      let mut arr = GPUDeviceArray4d::<f32>::zeros(shape, conn.clone());
      arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());
      arr
    })
  }
}

fn build_linear_normal<R: Rng + 'static>(x: Val<GPUDeviceOuterBatchArray1d<f32>>, std_dev: f32, src_ch: usize, dst_ch: usize, rng: Rc<RefCell<R>>, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
  let w = src(GPUDeviceArray2d::<f32>::normal_linear_init(std_dev, src_ch, dst_ch, rng));
  params.push_val(w.clone());
  let x = w.mult(x);
  x
}

fn build_linear(x: Val<GPUDeviceOuterBatchArray1d<f32>>, src_ch: usize, dst_ch: usize, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
  let w = src(GPUDeviceArray2d::<f32>::xavier_linear_init(src_ch, dst_ch, &mut thread_rng()));
  params.push_val(w.clone());
  let x = w.mult(x);
  x
}

fn build_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, conv_shape: Conv2dShape, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  let w = src(GPUDeviceArray4d::<f32>::kaiming_conv2d_init(
      conv_shape.ker_dims,
      conv_shape.src_features,
      conv_shape.features,
      &mut thread_rng()));
  params.push_val(w.clone());
  let x = w.conv(conv_shape.clone(), x);
  x
}

fn build_linearnet<R: Rng + 'static>(batch_sz: usize, rng: Rc<RefCell<R>>) -> (Val<GPUDeviceOuterBatchArray3d<u8>>, Val<GPUDeviceOuterBatchScalar<u32>>, Val<GPUDeviceOuterBatchArray1d<f32>>, Val<GPUDeviceScalar<f32>>, NodeVec) {
  let mut params = NodeVec::default();

  let image_var = src(GPUDeviceOuterBatchArray3d::<u8>::zeros_init(([28, 28, 1], batch_sz)));
  let label_var = src(GPUDeviceOuterBatchScalar::<u32>::zeros_init(batch_sz));

  let x = image_var.clone().dequantize(0.0_f32, 1.0_f32);

  let x = x.flatten();

  //let x = build_linear(x, 28 * 28 * 1, 10, &mut params);
  let x = build_linear_normal(x, 0.01, 28 * 28 * 1, 10, rng.clone(), &mut params);

  let logit_var = x.clone();

  let (nll, _) = x.softmax_categorical_nll(label_var.clone());
  let loss_var = nll.batch_sum();

  (image_var, label_var, logit_var, loss_var, params)
}

fn build_convnet(batch_sz: usize) -> (Val<GPUDeviceOuterBatchArray3d<u8>>, Val<GPUDeviceOuterBatchScalar<u32>>, Val<GPUDeviceOuterBatchArray1d<f32>>, Val<GPUDeviceScalar<f32>>, NodeVec) {
  let mut params = NodeVec::default();

  let image_var = src(GPUDeviceOuterBatchArray3d::<u8>::zeros_init(([28, 28, 1], batch_sz)));
  let label_var = src(GPUDeviceOuterBatchScalar::<u32>::zeros_init(batch_sz));

  let x = image_var.clone().dequantize(0.0_f32, 1.0_f32);

  let mut conv1 = Conv2dShape::default_nchw();
  conv1.src_dims = [28, 28];
  conv1.src_features = 1;
  conv1.ker_dims = [5, 5];
  conv1.features = 32;
  conv1.stride = [1, 1];
  conv1.zero_pad = [2, 2];

  let x = build_conv(x, conv1, &mut params);
  let x = x.positive_clip();

  let mut conv1_2 = Conv2dShape::default_nchw();
  conv1_2.src_dims = [28, 28];
  conv1_2.src_features = 32;
  conv1_2.ker_dims = [3, 3];
  conv1_2.features = 32;
  conv1_2.stride = [1, 1];
  conv1_2.zero_pad = [1, 1];

  let x = build_conv(x, conv1_2, &mut params);
  let x = x.positive_clip();

  let mut conv2_1 = Conv2dShape::default_nchw();
  conv2_1.src_dims = [28, 28];
  conv2_1.src_features = 32;
  conv2_1.ker_dims = [3, 3];
  conv2_1.features = 64;
  conv2_1.stride = [2, 2];
  conv2_1.zero_pad = [1, 1];

  let x = build_conv(x, conv2_1, &mut params);
  let x = x.positive_clip();

  let mut conv2_2 = Conv2dShape::default_nchw();
  conv2_2.src_dims = [14, 14];
  conv2_2.src_features = 64;
  conv2_2.ker_dims = [3, 3];
  conv2_2.features = 64;
  conv2_2.stride = [1, 1];
  conv2_2.zero_pad = [1, 1];

  let x = build_conv(x, conv2_2, &mut params);
  let x = x.positive_clip();

  let mut conv3_1 = Conv2dShape::default_nchw();
  conv3_1.src_dims = [14, 14];
  conv3_1.src_features = 64;
  conv3_1.ker_dims = [3, 3];
  conv3_1.features = 64;
  conv3_1.stride = [2, 2];
  conv3_1.zero_pad = [1, 1];

  let x = build_conv(x, conv3_1, &mut params);
  let x = x.positive_clip();

  let mut conv3_2 = Conv2dShape::default_nchw();
  conv3_2.src_dims = [7, 7];
  conv3_2.src_features = 64;
  conv3_2.ker_dims = [3, 3];
  conv3_2.features = 64;
  conv3_2.stride = [1, 1];
  conv3_2.zero_pad = [1, 1];

  let x = build_conv(x, conv3_2, &mut params);
  let x = x.positive_clip();

  let mut avg_pool = Pool2dShape::default_nchw();
  avg_pool.src_size = [7, 7];
  avg_pool.src_features = 64;
  avg_pool.ker_size = [7, 7];
  avg_pool.stride = [1, 1];
  //avg_pool.stride = [7, 7];
  avg_pool.zero_pad = [0, 0];

  let x = x.average_pool(avg_pool);
  let x = x.flatten();

  let x = build_linear(x, 1 * 1 * 64, 10, &mut params);

  let logit_var = x.clone();

  let (nll, _) = x.softmax_categorical_nll(label_var.clone());
  let loss_var = nll.batch_sum();

  (image_var, label_var, logit_var, loss_var, params)
}

fn main() {
  let mut group = DistProcGroup::default();
  for node in group {
    node.spawn(|proc| {
      println!("DEBUG: hello world: {}", proc.rank());

      let rng = Rc::new(RefCell::new(ReplayTapeRng::open(PathBuf::from("test_data/mnist_tape.bin"))));

      let mut dataset_cfg = MnistConfig::default();
      dataset_cfg.path = Some(PathBuf::from("../datasets/mnist"));

      let train_data = dataset_cfg.open_train_data().unwrap();
      let test_data = dataset_cfg.open_test_data();

      let num_classes = 10;

      //let batch_sz = 32;
      let batch_sz = 128;
      let batch_reps = 1;
      let display_interval = 1;

      let train_iter = {
        train_data
          .uniform_random_shared_rng(rng.clone())
          .batch_data(batch_sz)
      };

      let (image_var, label_var, logit_var, loss_var, params) = build_linearnet(batch_sz, rng.clone());
      //let (image_var, label_var, logit_var, loss_var, params) = build_convnet(batch_sz);

      let mut loss_sink = sink(loss_var.clone());
      let params = params.reversed();
      let grads = params.adjoints(&mut loss_sink);

      let batch_avg_rate = src_init(0.0_f32);
      let step_size = src_init(0.0_f32);
      let momentum = src_init(0.0_f32);

      let batch_grads_vec = grads.clone().vectorize();
      //let avg_grads_vec = zeros_like(batch_grads_vec.clone()).online_average(batch_avg_rate.clone(), batch_grads_vec.clone());
      let params_vec = params.clone().vectorize();
      let grad_vec_step = params_vec.clone().gradient_momentum_step(step_size.clone(), momentum.clone(), batch_grads_vec.clone());
      let params_devec = params_vec.clone().devectorize(params.clone());

      let init_txn = txn();
      params.persist(init_txn);
      let params_count = params.serialize_vec(init_txn, &mut ());
      println!("DEBUG: train: params len: {}", params.len());
      println!("DEBUG: train: params count: {}", params_count);
      assert_eq!(params_count, 10 * 28 * 28);
      println!("DEBUG: train: grads len: {}", grads.len());

      let mut params_h = MemArray1d::<f32>::zeros(params_count);
      let mut grads_h = MemArray1d::<f32>::zeros(params_count);
      assert_eq!(params_count, params.serialize_vec(init_txn, &mut params_h));
      println!("DEBUG: train: init: w: {:?}", &params_h.flat_view().unwrap().as_slice()[ .. 10]);
      println!("DEBUG: train: init: w: {:?}", &params_h.flat_view().unwrap().as_slice()[params_count - 10 .. ]);

      let mut stopwatch = Stopwatch::new();

      let mut image_data = MemArray4d::<u8>::zeros([28, 28, 1, batch_sz]);
      let mut label_data = MemArray1d::<u32>::zeros(batch_sz);
      let mut logit_data = MemArray2d::<f32>::zeros([num_classes, batch_sz]);
      let mut loss: f32 = 0.0;

      let mut labels = Vec::<u32>::with_capacity(batch_sz);
      let mut count = 0;
      let mut acc_ct = 0;
      let mut loss_sum: f32 = 0.0;

      for (batch_nr, batch) in train_iter.enumerate() {
        let iter_nr = batch_nr / batch_reps;
        let rep_nr = batch_nr % batch_reps;

        for (idx, (image, label)) in batch.into_iter().enumerate() {
          // TODO
          image_data.flat_view_mut().unwrap()
            .as_mut_slice()[idx * 28 * 28 * 1 .. (idx + 1) * 28 * 28 * 1]
            .copy_from_slice(image.flat_view().unwrap().as_slice());
          label_data.as_view_mut().as_mut_slice()[idx] = label;
        }

        let batch_txn = txn();
        image_var.deserialize(batch_txn, &mut image_data);
        label_var.deserialize(batch_txn, &mut label_data);
        params.persist(batch_txn);
        loss_var.eval(batch_txn);
        grads.eval(batch_txn);
        //batch_grads_vec.eval(batch_txn);
        //batch_avg_rate.set(batch_txn, 1.0 / (rep_nr + 1) as f32);
        //avg_grads_vec.eval(batch_txn);
        logit_var.serialize(batch_txn, &mut logit_data);
        loss_var.serialize(batch_txn, &mut loss);

        count += batch_sz;
        for idx in 0 .. batch_sz {
          let label = label_data.as_view().as_slice()[idx];
          let k = _arg_max(&logit_data.flat_view().unwrap().as_slice()[num_classes * idx .. num_classes * (idx + 1)]);
          if k == label as _ {
            acc_ct += 1;
          }
        }
        loss_sum += loss;

        if rep_nr == batch_reps - 1 {
          let step_txn = txn();
          step_size.set(step_txn, -0.003);
          //step_size.set(step_txn, -0.01 / batch_sz as f32);
          momentum.set(step_txn, 0.0);
          params.persist(step_txn);
          grads.persist(step_txn);
          batch_grads_vec.eval(step_txn);
          //avg_grads_vec.persist(step_txn);
          //assert_eq!(params_count, grads.serialize_vec(step_txn, &mut grads_h));
          assert_eq!(params_count, batch_grads_vec.serialize_vec(step_txn, &mut grads_h));
          assert!(grad_vec_step.eval(step_txn));

          let update_txn = txn();
          params_vec.persist(update_txn);
          assert!(params_devec.eval(update_txn));
          assert_eq!(params_count, params.serialize_vec(update_txn, &mut params_h));
        }

        if rep_nr == batch_reps - 1 && ((iter_nr + 1) % display_interval == 0 || iter_nr < display_interval) {
          println!("DEBUG: train: iters: {} acc: {:.4} ({}/{}) loss: {:.6} elapsed: {:.6} s",
              iter_nr + 1,
              acc_ct as f64 / count as f64,
              acc_ct,
              count,
              loss_sum / count as f32,
              stopwatch.click().lap_time());

          //println!("DEBUG: train:   logits: {:?}", &logit_data.flat_view().unwrap().as_slice()[ .. num_classes]);
          //println!("DEBUG: train:   logits: {:?}", &logit_data.flat_view().unwrap().as_slice()[(batch_sz - 1) * num_classes .. ]);
          //println!("DEBUG: train:   nll: {:?}", &logit_data.flat_view().unwrap().as_slice()[(batch_sz - 1) * num_classes .. ]);

          println!("DEBUG: train:   w: {:?}", &params_h.flat_view().unwrap().as_slice()[100 .. 110]);
          println!("DEBUG: train:   w: {:?}", &params_h.flat_view().unwrap().as_slice()[params_count - 110 .. params_count - 100]);
          println!("DEBUG: train:   dw: {:?}", &grads_h.flat_view().unwrap().as_slice()[100 .. 110]);
          println!("DEBUG: train:   dw: {:?}", &grads_h.flat_view().unwrap().as_slice()[params_count - 110 .. params_count - 100]);

          count = 0;
          acc_ct = 0;
          loss_sum = 0.0;
        }

        if (iter_nr + 1) >= 50 {
          break;
        }
        if rep_nr == batch_reps - 1 && (iter_nr + 1) * batch_sz >= 5 * 60000 {
          break;
        }
      }
    }).unwrap().join();
  }
}
