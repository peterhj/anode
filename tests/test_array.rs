extern crate memarray;

use memarray::*;
use memarray::linalg::*;

#[test]
fn test_view() {
  println!();
  let x = MemArray1d::<f32>::zeros(10);
  let y = MemArray2d::<f32>::zeros([10, 10]);
  let a = x.as_view().view(..);
  println!("DEBUG: a: {:?}", a.size());
  let b = x.as_view().view(5..);
  println!("DEBUG: b: {:?}", b.size());
  let c = x.as_view().view(1..8);
  println!("DEBUG: c: {:?}", c.size());
  let g = y.as_view().view(.., ..);
  println!("DEBUG: g: {:?}", g.size());
  let h = y.as_view().view(3.., ..8);
  println!("DEBUG: h: {:?}", h.size());
}

#[test]
fn test_linalg() {
  let w = MemArray2d::<f32>::zeros([10, 10]);
  let x = MemArray2d::<f32>::zeros([10, 10]);
  let mut y = MemArray2d::<f32>::zeros([10, 10]);
  y.as_view_mut().matrix_mult(w.as_view(), x.as_view());
}
