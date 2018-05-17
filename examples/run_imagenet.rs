extern crate anode;
extern crate colorimage;
extern crate rand;
extern crate stb_image;
extern crate superdata;

use colorimage::*;
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
  let mut iter = dataset.one_pass().map_data(|(value, _)| {
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
    maybe_image
  });

  let mut write_count = 0;
  for (idx, maybe_image) in iter.enumerate() {
    if maybe_image.is_none() {
      continue;
    }
    if thread_rng().gen_range(0, 1000) == 0 {
      let image = maybe_image.unwrap();
      let out_image = Image::new(image.width() as _, image.height() as _, 3, image.to_vec());
      let png_data = out_image.write_png().unwrap();
      let png_path = PathBuf::from(format!("tmp/test_{:04}.png", write_count));
      let mut png_file = File::create(&png_path).unwrap();
      png_file.write_all(&png_data).unwrap();
      println!("DEBUG: writing image: idx: {} out idx: {}", idx, write_count);
      write_count += 1;
    }
  }
}
