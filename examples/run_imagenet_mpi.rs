extern crate anode;
extern crate colorimage;
extern crate sharedmem;
extern crate superdata;

use anode::proc::*;
use colorimage::*;
use sharedmem::*;
use superdata::*;
use superdata::datasets::imagenet::*;

use std::path::{PathBuf};

struct JoinPartitionsMPI<Data> {
  all_len:  usize,
  part_len: usize,
  offset:   usize,
  data:     Data,
  proc:     DistProc,
}

impl<Data: RandomAccess> JoinPartitionsMPI<Data> {
  pub fn new(data: Data, proc: DistProc) -> Self {
    let mut all_len = [data.len() as u64];
    proc.barrier();
    proc.allreduce_sum(&mut all_len);
    proc.barrier();
    let all_len = all_len[0] as usize;
    let rdup_all_len = (all_len + proc.num_ranks() - 1) / proc.num_ranks() * proc.num_ranks();
    let rdup_part_len = rdup_all_len / proc.num_ranks();
    let offset = proc.rank() * rdup_part_len;
    println!("DEBUG: JoinPartitionsMPI: rank: {} all len: {} part len: {} offset: {} data len: {}",
        proc.rank(), all_len, rdup_part_len, offset, data.len());
    JoinPartitionsMPI{
      all_len:  all_len,
      part_len: rdup_part_len,
      offset:   offset,
      data:     data,
      proc:     proc,
    }
  }
}

impl<Data: RandomAccess<Item=(SharedMem<u8>, u32)>> RandomAccess for JoinPartitionsMPI<Data> {
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
}

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

      let val_partition = val_dataset.partition(proc.rank(), proc.num_ranks());
      println!("DEBUG: rank: {} val part len: {}", proc.rank(), val_partition.len());

      let train_partition = train_dataset.partition(proc.rank(), proc.num_ranks());
      println!("DEBUG: rank: {} train part len: {}", proc.rank(), train_partition.len());

      let batch_sz = 32;

      let train_iter =
          // TODO
          //train_partition
          JoinPartitionsMPI::new(train_partition, proc.clone())
          .one_pass()
          //.uniform_sample()
          .batch_data(batch_sz);

      for (iter_nr, mut batch) in train_iter.enumerate() {
        let maybe_image = match ColorImage::decode(&batch[0].0) {
          Ok(image) => {
            println!("DEBUG: image: dims: {} {}", image.width(), image.height());
            Some(image)
          }
          Err(_) => {
            println!("WARNING: image decode error");
            None
          }
        };
        break;
      }
    }).unwrap().join();
  }
}
