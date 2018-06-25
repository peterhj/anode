extern crate anode;
//extern crate cuda_dnn;
extern crate gpudevicemem;
extern crate memarray;
extern crate rand;
extern crate superdata;

use anode::*;
use anode::ops::*;
use anode::ops_gpu::*;
use anode::utils::*;
//use cuda_dnn::*;
use gpudevicemem::array::*;
use memarray::*;
use superdata::*;
use superdata::datasets::cifar10::*;

use rand::*;
use std::env;
use std::fs::{File};
use std::io::{Write};
use std::path::{PathBuf};

fn build_resnet(batch_sz: usize) -> (Val<GPUDeviceOuterBatchArray3d<u8>>, Val<GPUDeviceOuterBatchScalar<u32>>, Val<bool>, Val<f32>, NodeVec, NodeVec, NodeVec) {
  let mut params = NodeVec::default();
  let mut online_stats = NodeVec::default();
  let mut avg_stats = NodeVec::default();

  let image_var = zeros(([28, 28, 3], batch_sz));
  let label_var = zeros(batch_sz);
  let online = src_init(false);
  let avg_rate = src_init(0.0);
  let epsilon: f32 = 1.0e-6;

  let x = image_var.clone().dequantize(0.0_f32, 1.0_f32);

  let mut conv1 = Conv2dShape::default_nchw();
  conv1.src_size = [28, 28, 3];
  conv1.ker_size = [3, 3];
  conv1.features = 16;
  conv1.stride = [1, 1];
  conv1.zero_pad = [1, 1];
  let w1 = touch(GPUDeviceArray4d::<f32>::normal_init([3, 3, 3, 16], 0.0, 0.01, &mut thread_rng()));
  let b1 = touch(GPUDeviceArray1d::<f32>::zeros_init(16));
  params.push_val(w1.clone());
  params.push_val(b1.clone());
  //let x = Conv2dAffineOp::build_device_obatch_val(conv1, w1, x, b1);
  let x = w1.conv_add(conv1, x, b1);
  let (x, x_mean, x_var, x_avg_mean, x_avg_var) = x.batch_normalize_2d([0, 1], online.clone(), avg_rate.clone(), epsilon);
  online_stats.push_val(x_mean);
  online_stats.push_val(x_var);
  avg_stats.push_val(x_avg_mean);
  avg_stats.push_val(x_avg_var);
  let x = x.rect();

  // TODO

  (image_var, label_var, online, avg_rate, params, online_stats, avg_stats)
}

fn main() {
  let args: Vec<_> = env::args().collect();
  let train_path = PathBuf::from(&args[1]);
  let data_cfg = Cifar10Config{
    train_data: Some(train_path),
    .. Default::default()
  };
  let train_data = Cifar10Data::open_train(data_cfg).unwrap();

  let batch_sz = 128;
  let num_classes = 10;

  let train_iter = train_data.uniform_random(&mut thread_rng())
    .batch_data(batch_sz)
    .map_data(|items: Vec<_>| {
      // TODO: pad first, then crop.
      let mut image_batch = MemArray4d::<u8>::zeros([28, 28, 3, batch_sz]);
      //let mut image_batch = MemArray4d::<u8>::zeros([32, 32, 3, batch_sz]);
      let mut label_batch = MemArray1d::<u32>::zeros(batch_sz);
      for (idx, &(ref value, label)) in items.iter().enumerate() {
        let off_x = thread_rng().gen_range(0, 4);
        let off_y = thread_rng().gen_range(0, 4);
        for c in 0 .. 3 {
          for y in 0 .. 28 {
            let src_off = off_x + 32 * (off_y + y + 32 * (c + 3 * idx));
            image_batch.as_view_mut()
              .view_mut(.., y .. y + 1, c .. c + 1, idx .. idx + 1)
              .flat_slice_mut().unwrap()
              .copy_from_slice(&value[src_off .. src_off + 28]);
          }
        }
        label_batch.as_view_mut()
          .flat_slice_mut().unwrap()[idx] = label;
      }
      (image_batch, label_batch)
    });

  // TODO
  let (image_var, label_var, /*logit_var, loss_var,*/ online_var, avg_rate_var, params, online_stats, avg_stats) = build_resnet(batch_sz);
  let mut loss: Sink = { unimplemented!() };
  //let mut loss = Sink::from(loss_var);
  let params = params.reversed();
  let grads = params.adjoints(&mut loss);

  let mut image_batch = MemArray4d::<u8>::zeros([32, 32, 3, batch_sz]);
  let mut label_batch = MemArray1d::<u32>::zeros(batch_sz);
  let mut logit_batch = MemArray2d::<f32>::zeros([num_classes, batch_sz]);

  for (iter_nr, (mut image_batch, mut label_batch)) in train_iter.enumerate() {
    // TODO: do preprocessing here.

    // TODO
    let batch = txn();
    params.persist(batch);
    image_var.deserialize(batch, &mut image_batch);
    label_var.deserialize(batch, &mut label_batch);
    online_var.set(batch, true);
    avg_rate_var.set(batch, if iter_nr == 0 { 1.0 } else { 0.003 });
    /*
    logit_var.eval(batch);
    grads.eval(batch);

    // TODO: log training statistics.
    let mut loss_batch: f32 = 0.0;
    logit_var.serialize(batch, &mut logit_batch);
    loss_var.serialize(batch, &mut loss_batch);

    let step = txn();
    grads.persist(step);
    online_stats.persist(step);
    sgd.step(step, params, grads);
    avg_stats.eval(step);
    */
  }
}
