#![feature(alloc_system)]

extern crate alloc_system;

//extern crate anode;
extern crate colorimage;
//extern crate gpudevicemem;
//extern crate memarray;
//extern crate rand;
//extern crate sharedmem;
extern crate superdata;

use colorimage::*;
use superdata::*;
use superdata::datasets::cityscapes::*;

use std::path::{PathBuf};

fn main() {
  let val_path = PathBuf::from("/rscratch18/phj/data/cityscapes/cityscapes_fine_val.tar");
  let val_dataset = CityscapesTarData::open(val_path).unwrap();

  let train_path = PathBuf::from("/rscratch18/phj/data/cityscapes/cityscapes_fine_train.tar");
  let train_dataset = CityscapesTarData::open(train_path).unwrap();

  for (idx, (image_png, label_png)) in val_dataset.one_pass().enumerate() {
    let image = match ColorImage::decode(&image_png) {
      Ok(im) => {
        println!("DEBUG: image: {} width: {} height: {}",
            idx, im.width(), im.height());
        Some(im)
      }
      Err(_) => {
        println!("WARNING: image decode error: {}", idx);
        None
      }
    };
    let label = match ColorImage::decode(&label_png) {
      Ok(im) => {
        println!("DEBUG: label: {} width: {} height: {}",
            idx, im.width(), im.height());
        Some(im)
      }
      Err(_) => {
        println!("WARNING: image decode error: {}", idx);
        None
      }
    };
  }

  for (idx, (image_png, label_png)) in train_dataset.one_pass().enumerate() {
    let image = match ColorImage::decode(&image_png) {
      Ok(im) => {
        println!("DEBUG: image: {} width: {} height: {}",
            idx, im.width(), im.height());
        Some(im)
      }
      Err(_) => {
        println!("WARNING: image decode error: {}", idx);
        None
      }
    };
    let label = match ColorImage::decode(&label_png) {
      Ok(im) => {
        println!("DEBUG: label: {} width: {} height: {}",
            idx, im.width(), im.height());
        Some(im)
      }
      Err(_) => {
        println!("WARNING: image decode error: {}", idx);
        None
      }
    };
  }
}
