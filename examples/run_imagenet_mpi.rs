extern crate anode;
extern crate colorimage;
extern crate gpudevicemem;
extern crate memarray;
#[cfg(feature = "mpi")] extern crate mpich;
extern crate rand;
extern crate sharedmem;
extern crate superdata;

use anode::*;
use anode::ops::*;
#[cfg(feature = "mpi")] use anode::proc::*;
use anode::utils::*;
use colorimage::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
#[cfg(feature = "mpi")] use mpich::*;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform, Normal};
//use rand::rngs::mock::*;
use sharedmem::*;
use superdata::*;
use superdata::datasets::imagenet::*;
use superdata::utils::*;

use std::cmp::{max, min};
use std::path::{PathBuf};
use std::rc::{Rc};

pub trait XavierLinearInit<Shape, T, R: Rng> {
  type RValue;

  fn xavier_linear_init(shape: Shape, src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

pub trait XavierConv2dInit<Shape, T, R: Rng> {
  type RValue;

  fn xavier_conv2d_init(shape: Shape, ker_sz: [usize; 2], src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

pub trait KaimingConv2dInit<Shape, T, R: Rng> {
  type RValue;

  fn kaiming_conv2d_init(shape: Shape, ker_sz: [usize; 2], src: usize, dst: usize, seed_rng: &mut R) -> Self::RValue;
}

impl<R: Rng> XavierLinearInit<[usize; 2], f32, R> for GPUDeviceArray2d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn xavier_linear_init(shape: [usize; 2], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
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

impl<R: Rng> XavierConv2dInit<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn xavier_conv2d_init(shape: [usize; 4], ker_sz: [usize; 2], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
      // TODO: seed the local rng here.
      let mut h_arr = MemArray4d::<f32>::zeros(shape);
      {
        // TODO: distribution.
        let half_width = (6.0 / (ker_sz[0] * ker_sz[1] * (src_ch + dst_ch)) as f64).sqrt();
        let dist = Uniform::new_inclusive(-half_width, half_width);
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

impl<R: Rng> KaimingConv2dInit<[usize; 4], f32, R> for GPUDeviceArray4d<f32> {
  type RValue = Rc<Fn(Txn, GPUDeviceConn) -> Self>;

  fn kaiming_conv2d_init(shape: [usize; 4], ker_sz: [usize; 2], src_ch: usize, dst_ch: usize, seed_rng: &mut R) -> Self::RValue {
    //let seed = seed_rng.next_u64();
    Rc::new(move |_, conn: GPUDeviceConn| {
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

fn build_linear(x: Val<GPUDeviceOuterBatchArray1d<f32>>, src_ch: usize, dst_ch: usize, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
  let w = src(GPUDeviceArray2d::<f32>::xavier_linear_init([dst_ch, src_ch], src_ch, dst_ch, &mut thread_rng()));
  let b = src(GPUDeviceArray1d::<f32>::zeros_init(dst_ch));
  params.push_val(w.clone());
  params.push_val(b.clone());
  let x = w.mult_add(x, b);
  x
}

fn build_batch_norm_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: TCell<bool>, avg_rate: TCell<f32>, conv_shape: Conv2dShape, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  let w = src(GPUDeviceArray4d::<f32>::kaiming_conv2d_init(
      [conv_shape.ker_size[0], conv_shape.ker_size[1], conv_shape.src_size[2], conv_shape.features],
      conv_shape.ker_size,
      conv_shape.src_size[2],
      conv_shape.features,
      &mut thread_rng()));
  let b = src(GPUDeviceArray1d::<f32>::zeros_init(conv_shape.features));
  params.push_val(w.clone());
  params.push_val(b.clone());
  let x = w.conv_add(conv_shape.clone(), x, b);
  let epsilon = 1.0e-6_f32;
  let (x, x_mean, x_var, x_avg_mean, x_avg_var) = x.batch_normalize_2d([0, 1], online, avg_rate, epsilon);
  online_stats.push_val(x_mean);
  online_stats.push_val(x_var);
  avg_stats.push_val(x_avg_mean);
  avg_stats.push_val(x_avg_var);
  x
}

fn build_residual3_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: TCell<bool>, avg_rate: TCell<f32>, conv_shape: Conv2dShape, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  // TODO
  let mut conv1 = conv_shape;
  let y = build_batch_norm_conv(x.clone(), online.clone(), avg_rate.clone(), conv1, params, online_stats, avg_stats);
  let y = y.positive_clip();
  // TODO
  let mut conv2 = conv_shape;
  let y = build_batch_norm_conv(y, online.clone(), avg_rate.clone(), conv2, params, online_stats, avg_stats);
  let y = y.positive_clip();
  // TODO
  let mut conv3 = conv_shape;
  let y = build_batch_norm_conv(y, online.clone(), avg_rate.clone(), conv3, params, online_stats, avg_stats);
  let y = y + x;
  let y = y.positive_clip();
  y
}

fn build_proj_residual3_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, online: TCell<bool>, avg_rate: TCell<f32>, conv_shape: Conv2dShape, src_size: [usize; 2], src_features: usize, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  // TODO
  unimplemented!();
}

fn build_resnet(batch_sz: usize) -> (Val<GPUDeviceOuterBatchArray3d<u8>>, Val<GPUDeviceOuterBatchScalar<u32>>, Val<GPUDeviceOuterBatchArray1d<f32>>, Val<GPUDeviceScalar<f32>>, TCell<bool>, TCell<f32>, NodeVec, NodeVec, NodeVec) {
  let mut params = NodeVec::default();
  let mut online_stats = NodeVec::default();
  let mut avg_stats = NodeVec::default();

  let image_var = src(GPUDeviceOuterBatchArray3d::<u8>::zeros_init(([224, 224, 3], batch_sz)));
  let label_var = src(GPUDeviceOuterBatchScalar::<u32>::zeros_init(batch_sz));
  let online = TCell::new(false);
  let avg_rate = TCell::new(0.0_f32);
  let epsilon: f32 = 1.0e-6;

  let x = image_var.clone().dequantize(0.0_f32, 1.0_f32);

  let mut conv1 = Conv2dShape::default_nchw();
  conv1.src_size = [224, 224, 3];
  conv1.ker_size = [3, 3];
  conv1.features = 64;
  conv1.stride = [2, 2];
  conv1.zero_pad = [1, 1];

  let x = build_batch_norm_conv(x, online.clone(), avg_rate.clone(), conv1, &mut params, &mut online_stats, &mut avg_stats);
  let x = x.positive_clip();

  let mut pool1 = Pool2dShape::default_nchw();
  pool1.src_size = [112, 112];
  pool1.src_features = 64;
  pool1.ker_size = [3, 3];
  pool1.stride = [2, 2];
  pool1.zero_pad = [1, 1];

  let x = x.max_pool(pool1);

  let mut conv2 = Conv2dShape::default_nchw();
  conv2.src_size = [56, 56, 128];
  conv2.ker_size = [3, 3];
  conv2.features = 128;
  conv2.stride = [1, 1];
  conv2.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, online.clone(), avg_rate.clone(), conv2, [56, 56], 64, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. 3 {
    x = build_residual3_conv(x, online.clone(), avg_rate.clone(), conv2, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv3 = Conv2dShape::default_nchw();
  conv3.src_size = [28, 28, 256];
  conv3.ker_size = [3, 3];
  conv3.features = 256;
  conv3.stride = [1, 1];
  conv3.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, online.clone(), avg_rate.clone(), conv3, [56, 56], 128, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. 4 {
    x = build_residual3_conv(x, online.clone(), avg_rate.clone(), conv3, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv4 = Conv2dShape::default_nchw();
  conv4.src_size = [14, 14, 512];
  conv4.ker_size = [3, 3];
  conv4.features = 512;
  conv4.stride = [1, 1];
  conv4.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, online.clone(), avg_rate.clone(), conv4, [28, 28], 256, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. 6 {
    x = build_residual3_conv(x, online.clone(), avg_rate.clone(), conv4, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv5 = Conv2dShape::default_nchw();
  conv5.src_size = [7, 7, 512];
  conv5.ker_size = [3, 3];
  conv5.features = 512;
  conv5.stride = [1, 1];
  conv5.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, online.clone(), avg_rate.clone(), conv5, [14, 14], 512, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. 3 {
    x = build_residual3_conv(x, online.clone(), avg_rate.clone(), conv5, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut avg_pool = Pool2dShape::default_nchw();
  pool1.src_size = [7, 7];
  pool1.src_features = 2048;
  pool1.ker_size = [7, 7];
  pool1.stride = [7, 7];
  pool1.zero_pad = [0, 0];

  let x = x.average_pool(avg_pool);
  let x = x.flatten();

  let x = build_linear(x, 2048, 1000, &mut params);

  let logit_var = x.clone();

  let (nll, _) = x.softmax_categorical_nll(label_var.clone());
  let loss_var = nll.batch_sum();

  (image_var, label_var, logit_var, loss_var, online, avg_rate, params, online_stats, avg_stats)
}

#[cfg(not(feature = "mpi"))]
fn main() {
}

#[cfg(feature = "mpi")]
fn main() {
  let mut group = DistProcGroup::default();
  for node in group {
    node.spawn(|proc| {
      println!("DEBUG: hello world: {}", proc.rank());

      let train_data_path = PathBuf::from("/scratch/snx3000/peterhj/data/ilsvrc2012/ILSVRC2012_img_train.tar");
      let val_data_path = PathBuf::from("/scratch/snx3000/peterhj/data/ilsvrc2012/ILSVRC2012_img_val.tar");

      let data_cfg = ImagenetConfig{
        train_data: Some(train_data_path),
        //wordnet_ids: Some(wordnet_ids_path),
        val_data: Some(val_data_path),
        //val_ground_truth: Some(val_ground_truth_path),
        .. Default::default()
      };

      println!("DEBUG: loading val data...");
      let val_dataset = ImagenetValData::load_index(data_cfg.clone()).unwrap();
      println!("DEBUG: val len: {}", val_dataset.len());

      println!("DEBUG: loading train data...");
      let train_dataset = ImagenetTrainData::load_index(data_cfg.clone()).unwrap();
      println!("DEBUG: train len: {}", train_dataset.len());

      /*let val_shard = val_dataset.partition(proc.rank(), proc.num_ranks());
      println!("DEBUG: rank: {} val part len: {}", proc.rank(), val_shard.len());

      let train_shard = train_dataset.partition(proc.rank(), proc.num_ranks());
      println!("DEBUG: rank: {} train part len: {}", proc.rank(), train_shard.len());*/

      let num_data_workers = 20;
      let num_classes = 1000;
      let batch_sz = 32;
      let eval_interval = 5000;

      // TODO: sharding.
      let train_iter = {
          let mut worker_rngs = Vec::with_capacity(num_data_workers);
          for _ in 0 .. num_data_workers {
            worker_rngs.push(SmallRng::from_rng(thread_rng()).unwrap());
          }
          async_join_data(num_data_workers, |worker_rank| {
            let mut rng = worker_rngs[worker_rank].clone();
            train_dataset.clone()
              .uniform_random(&mut rng)
              .map_data(|(value, label)| {
                let maybe_image = match ColorImage::decode(&value) {
                  Ok(mut image) => {
                    //println!("DEBUG: image: dims: {} {}", image.width(), image.height());
                    // TODO: data augmentation.
                    image.resize(224, 224);
                    Some(image)
                  }
                  Err(_) => {
                    println!("WARNING: image decode error");
                    None
                  }
                };
                (maybe_image, label)
              })
          })
          .async_prefetch_data(batch_sz * max(2, num_data_workers))
          .batch_data(batch_sz)
      };

      let mut stopwatch = Stopwatch::new();
      let mut images = Vec::with_capacity(batch_sz);
      let mut labels = Vec::with_capacity(batch_sz);

      let mut image_batch = MemArray4d::<u8>::zeros([224, 224, 3, batch_sz]);
      let mut label_batch = MemArray1d::<u32>::zeros(batch_sz);
      let mut logit_batch = MemArray2d::<f32>::zeros([num_classes, batch_sz]);
      let mut loss: f32 = 0.0;

      for (iter_nr, batch) in train_iter.enumerate() {
        images.clear();
        labels.clear();
        for (idx, (maybe_image, label)) in batch.into_iter().enumerate() {
          images.push(maybe_image);
          labels.push(label);
          label_batch.as_view_mut().as_mut_slice()[idx] = label;
        }

        let batch_txn = txn();
        // TODO: evaluate the batch.
        /*
        image_var.deserialize(batch_txn, &mut image_batch);
        label_var.deserialize(batch_txn, &mut label_batch);
        online_var.propose(batch_txn, |_| true);
        loss_var.eval(batch_txn);
        grads.eval(batch_txn);
        logit_var.serialize(batch_txn, &mut logit_batch);
        loss_var.serialize(batch_txn, &mut loss);
        */

        let mut acc_ct = 0;
        for idx in 0 .. batch_sz {
          let k = _arg_max(&logit_batch.flat_view().unwrap().as_slice()[num_classes * idx .. num_classes * (idx + 1)]);
          if k == labels[idx] as _ {
            acc_ct += 1;
          }
        }

        let step_txn = txn();
        // TODO: gradient step.
        // TODO: update moving average stats.

        if (iter_nr + 1) % 100 == 0 {
          println!("DEBUG: train: iters: {} acc: {:.4} ({}/{}) loss: {:.6} elapsed: {:.6} s",
              iter_nr + 1,
              acc_ct as f64 / batch_sz as f64,
              acc_ct, batch_sz,
              loss,
              stopwatch.click().lap_time());
        }
        continue;

        if (iter_nr + 1) % eval_interval == 0 {
          // TODO: validation.
          let val_shard = val_dataset.partition(proc.rank(), proc.num_ranks());
          let shard_len = val_shard.len();
          let val_iter =
              val_shard
              .one_pass()
              .round_up_data(batch_sz)
              .batch_data(batch_sz);
          let mut shard_ctr = 0;
          'eval_batch_loop: for eval_batch in val_iter {
            let batch_txn = txn();
            // TODO: evaluate the batch.
            for idx in 0 .. batch_sz {
              // TODO: calculate accuracy stats.
              shard_ctr += 1;
              if shard_ctr == shard_len {
                break 'eval_batch_loop;
              }
            }
          }
        }
      }
    }).unwrap().join();
  }
}
