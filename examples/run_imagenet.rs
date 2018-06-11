extern crate anode;
extern crate colorimage;
extern crate memarray;
extern crate rand;
extern crate stb_image;
extern crate superdata;

use colorimage::*;
use memarray::*;
use stb_image::image::{Image};
use superdata::*;
use superdata::datasets::imagenet::*;

use rand::*;
use std::env;
use std::fs::{File};
use std::io::{Write};
use std::path::{PathBuf};

fn main() {
  let args: Vec<_> = env::args().collect();
  let train_path = PathBuf::from(&args[1]);
  let wordnet_ids_path = PathBuf::from(&args[2]);
  let datacfg = ImagenetConfig{
    train_data: Some(train_path),
    wordnet_ids: Some(wordnet_ids_path),
    .. Default::default()
  };
  let dataset = ImagenetTrainData::open(datacfg).unwrap();

  let mut count = 0;
  let iter = dataset.one_pass().map_data(|(value, label)| {
    let mut image = ColorImage::new();
    let maybe_image = match decode_image(&value, &mut image) {
      Ok(_) => {
        println!("DEBUG: image idx: {} dims: {} {}", count, image.width(), image.height());
        Some(image)
      }
      Err(_) => {
        println!("WARNING: image idx: {} decode error", count);
        None
      }
    };
    count += 1;
    (maybe_image, label)
  });

  let mut write_count = 0;
  for (idx, (maybe_image, _)) in iter.enumerate() {
    if maybe_image.is_none() {
      continue;
    }

    let image = maybe_image.unwrap();
    let im_sz = [image.width() as _, image.height() as _, image.channels() as _];
    let mut im_arr = MemArray3d::zeros(im_sz);
    image.dump_planes(im_arr.flat_view_mut().unwrap().as_mut_slice());

    if thread_rng().gen_range(0, 1000) == 0 {
      let image_sz = image.width() as usize * image.height() as usize * 3;
      let mut image_buf = Vec::with_capacity(image_sz);
      for _ in 0 .. image_sz {
        image_buf.push(0);
      }
      image.dump_pixels(&mut image_buf);
      let out_image = Image::new(image.width() as _, image.height() as _, 3, image_buf);
      let png_data = out_image.write_png().unwrap();
      let png_path = PathBuf::from(format!("tmp/test_{:04}_{:08}.png", write_count, idx));
      let mut png_file = File::create(&png_path).unwrap();
      png_file.write_all(&png_data).unwrap();
      println!("DEBUG: writing image: idx: {} out idx: {}", idx, write_count);
      write_count += 1;
    }
  }
}
