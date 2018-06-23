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
use gpudevicemem::array::*;
use memarray::*;
#[cfg(feature = "mpi")] use mpich::*;
use rand::prelude::*;
use rand::rngs::mock::*;
use sharedmem::*;
use superdata::*;
use superdata::datasets::imagenet::*;

use std::cmp::{max, min};
use std::path::{PathBuf};

/*struct SplitShardsMPI<Data> {
  part_len: usize,
  part_off: usize,
  all_data: Data,
  proc:     DistProc,
}

impl<Data: RandomAccess> SplitShardsMPI<Data> {
  pub fn new(all_data: Data, proc: DistProc) -> Self {
    //let mut all_len = [data.len() as u64];
    //proc.barrier();
    //proc.allreduce_sum(&mut all_len);
    //proc.barrier();
    //let all_len = all_len[0] as usize;
    let rdup_all_len = (all_data.len() + proc.num_ranks() - 1) / proc.num_ranks() * proc.num_ranks();
    let rdup_part_len = rdup_all_len / proc.num_ranks();
    let part_off = proc.rank() * rdup_part_len;
    let part_len = min(all_data.len(), (proc.rank() + 1) * rdup_part_len) - part_off;
    println!("DEBUG: SplitShardsMPI: rank: {} all len: {} part offset: {} part len: {}",
        proc.rank(), all_data.len(), part_off, part_len);
    //let rm_buf = MPIWindow::new(64 * 1024 * 1024, &mut MPIComm::world()).unwrap();
    //println!("DEBUG:   rma window created");
    SplitShardsMPI{
      part_len: part_len,
      part_off: part_off,
      all_data: all_data,
      proc:     proc,
    }
  }
}

struct JoinShardsMPI<Data> {
  all_len:  usize,
  part_len: usize,
  offset:   usize,
  data:     Data,
  proc:     DistProc,
  //rm_buf:   MPIWindow<u8>,
}

impl<Data: RandomAccess> JoinShardsMPI<Data> {
  pub fn new(data: Data, proc: DistProc) -> Self {
    let mut all_len = [data.len() as u64];
    proc.barrier();
    proc.allreduce_sum(&mut all_len);
    proc.barrier();
    let all_len = all_len[0] as usize;
    let rdup_all_len = (all_len + proc.num_ranks() - 1) / proc.num_ranks() * proc.num_ranks();
    let rdup_part_len = rdup_all_len / proc.num_ranks();
    let offset = proc.rank() * rdup_part_len;
    println!("DEBUG: JoinShardsMPI: rank: {} all len: {} part len: {} offset: {} data len: {}",
        proc.rank(), all_len, rdup_part_len, offset, data.len());
    //let rm_buf = MPIWindow::new(64 * 1024 * 1024, &mut MPIComm::world()).unwrap();
    //println!("DEBUG:   rma window created");
    JoinShardsMPI{
      all_len:  all_len,
      part_len: rdup_part_len,
      offset:   offset,
      data:     data,
      proc:     proc,
    }
  }
}

impl<Data: RandomAccess<Item=(SharedMem<u8>, u32)>> RandomAccess for JoinShardsMPI<Data> {
  type Item = <Data as RandomAccess>::Item;

  fn len(&self) -> usize {
    self.all_len
  }

  fn at(&self, idx: usize) -> Self::Item {
    let src_rank = idx / self.part_len;
    let src_idx = idx % self.part_len;
    if src_rank == self.proc.rank() {
      self.data.at(src_idx)
    } else {
      // TODO
      self.data.at(src_idx % self.data.len())
      //unimplemented!();
      /*
      let mut dst_buf = ...;
      */
    }
  }
}*/

fn build_linear(x: Val<GPUDeviceOuterBatchArray1d<f32>>, src: usize, dst: usize, params: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray1d<f32>> {
  // TODO
  unimplemented!();
}

fn build_batch_norm_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, conv_shape: Conv2dShape, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  // TODO
  unimplemented!();
}

fn build_residual3_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, conv_shape: Conv2dShape, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
  // TODO
  unimplemented!();
}

fn build_proj_residual3_conv(x: Val<GPUDeviceOuterBatchArray3d<f32>>, conv_shape: Conv2dShape, src_size: [usize; 2], src_features: usize, params: &mut NodeVec, online_stats: &mut NodeVec, avg_stats: &mut NodeVec) -> Val<GPUDeviceOuterBatchArray3d<f32>> {
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

  let x = build_batch_norm_conv(x, conv1, &mut params, &mut online_stats, &mut avg_stats);
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
  x = build_proj_residual3_conv(x, conv2, [56, 56], 64, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. 2 {
    x = build_residual3_conv(x, conv2, &mut params, &mut online_stats, &mut avg_stats);
  }

  let mut conv3 = Conv2dShape::default_nchw();
  conv3.src_size = [28, 28, 256];
  conv3.ker_size = [3, 3];
  conv3.features = 256;
  conv3.stride = [1, 1];
  conv3.zero_pad = [0, 0];

  let mut x = x;
  x = build_proj_residual3_conv(x, conv3, [56, 56], 128, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. 2 {
    x = build_residual3_conv(x, conv3, &mut params, &mut online_stats, &mut avg_stats);
  }

  // FIXME
  /*
  let mut x = x;
  x = build_proj_residual3_conv(x, conv4, [28, 28], 256, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. _ {
    x = build_residual3_conv(x, conv4, &mut params, &mut online_stats, &mut avg_stats);
  }
  */

  // FIXME
  /*
  let mut x = x;
  x = build_proj_residual3_conv(x, conv5, [14, 14], 512, &mut params, &mut online_stats, &mut avg_stats);
  for _ in 1 .. _ {
    x = build_residual3_conv(x, conv5, &mut params, &mut online_stats, &mut avg_stats);
  }
  */

  let mut avg_pool = Pool2dShape::default_nchw();
  pool1.src_size = [7, 7];
  pool1.src_features = 512;
  pool1.ker_size = [7, 7];
  pool1.stride = [7, 7];
  pool1.zero_pad = [0, 0];

  let x = x.average_pool(avg_pool);
  let x = x.flatten();

  // FIXME
  /*
  let x = build_linear(x, 512, 1000, &mut params);
  */

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
