extern crate anode;
extern crate memarray;
extern crate rand;
extern crate superdata;

use anode::*;
use memarray::*;
use superdata::*;
use superdata::datasets::cifar10::*;

use rand::*;
use std::env;
use std::fs::{File};
use std::io::{Write};
use std::path::{PathBuf};

fn main() {
  let args: Vec<_> = env::args().collect();
  let train_path = PathBuf::from(&args[1]);
  let data_cfg = Cifar10Config{
    train_data: Some(train_path),
    .. Default::default()
  };
  let train_data = Cifar10Data::open_train(data_cfg).unwrap();

  let batch_sz = 128;

  let train_iter = train_data.uniform_random(&mut thread_rng())
    .batch_data(batch_sz)
    .map_data(|items: Vec<_>| {
      let mut image_batch = MemArray4d::<u8>::zeros([28, 28, 3, batch_sz]);
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

  let online_var = TCell::new(false);

  // TODO

  for (iter_nr, (image_batch, label_batch)) in train_iter.enumerate() {
    // TODO
    let t = txn();
    online_var.propose(t, |_| true);
    /*
    image_var.deserialize(t, &mut image_batch);
    label_var.deserialize(t, &mut label_batch);
    grads.eval(t);
    sgd.step(t, params, grads);
    */
  }
}
