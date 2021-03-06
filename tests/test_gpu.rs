extern crate anode;
extern crate gpudevicemem;
extern crate memarray;
extern crate rand;

use anode::*;
use anode::context::*;
use anode::log::*;
use anode::ops::*;
use anode::ops_gpu::*;
use anode::utils::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
use memarray::*;
use rand::*;

use std::rc::{Rc};
use std::thread::{sleep};
use std::time::{Duration};

/*#[test]
fn test_stream() {
  println!();
  let stream = GPUDeviceStreamPool::new(GPUDeviceId(0));
}*/

#[test]
#[should_panic]
fn test_gpu_src_eval_fail() {
  println!();
  let x: Val<_> = src(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  x.eval(txn());
}

#[test]
fn test_gpu_random_src_eval() {
  println!();
  let x: Val<_> = random_bits(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<u32>::zeros(1024, conn)
  }));
  let t = txn();
  x.eval(t);
  x.reset();
  x.eval(t);
  let t = txn();
  x.eval(t);
  x.release();
  x.eval(t);
  let t = txn();
  x.eval(t);
  x.reset();
  x.eval(t);
}

#[test]
fn test_gpu_zeros_eval() {
  println!();

  let x: Val<_> = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    let mut h_arr = MemArray1d::<f32>::zeros(1024);
    {
      let mut h_arr = h_arr.as_view_mut();
      let mut x = h_arr.flat_slice_mut().unwrap();
      for k in 0 .. 1024 {
        x[k] = k as f32;
      }
    }
    let mut arr = GPUDeviceArray1d::<f32>::zeros(1024, conn.clone());
    arr.as_view_mut().sync_copy_mem(h_arr.as_view(), conn.clone());

    let mut h_arr2 = MemArray1d::<f32>::zeros(1024);
    arr.as_view().sync_dump_mem(h_arr2.as_view_mut(), conn.clone());
    {
      let mut h_arr2 = h_arr2.as_view();
      let x = h_arr2.flat_slice().unwrap();
      for k in 0 .. 1024 {
        assert_eq!(x[k], k as f32);
      }
    }

    arr
  }));

  let t = txn();
  x.eval(t);
  x.eval(t);
  x.eval(t);

  {
    let ctx = thread_ctx().gpu();
    let mut stream = ctx.pool();
    let conn = stream.conn();
    let section = GPUAsyncSection::new(conn.clone());
    let guard = section.push(conn.clone());
    //let x = x.value();
    let v = x.get(t);
    let mut h_arr = MemArray1d::<f32>::zeros(1024);
    v.as_view().sync_dump_mem(h_arr.as_view_mut(), conn.clone());
    {
      let mut h_arr = h_arr.as_view();
      let x = h_arr.flat_slice().unwrap();
      for k in 0 .. 1024 {
        assert_eq!(x[k], 0.0);
      }
    }
  }

  /*println!("DEBUG: sleeping...");
  sleep(Duration::from_secs(2));
  println!("DEBUG: done sleeping");*/
}

#[test]
fn test_gpu_zeros_init_uniform() {
  println!();
  let x: Val<_> = touch(GPUDeviceArray1d::<f32>::uniform_init(1024, -1.0, 1.0, &mut thread_rng()));
  let t = txn();
  x.eval(t);
  {
    let ctx = thread_ctx().gpu();
    let mut stream = ctx.pool();
    let conn = stream.conn();
    let section = GPUAsyncSection::new(conn.clone());
    let guard = section.push(conn.clone());
    let y = x.get(t);
    let mut z = MemArray1d::<f32>::zeros(1024);
    println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
    y.flat_view().unwrap().sync_dump_mem(z.as_view_mut(), conn.clone());
    println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
  }
}

#[test]
fn test_gpu_io_deserialize() {
  println!();
  let x: Val<_> = src(GPUDeviceOuterBatchArray3d::<f32>::zeros_init(([32, 32, 3], 64)));
  let mut src = MemArray4d::<f32>::zeros([32, 32, 3, 64]);
  let t = txn();
  x.deserialize(t, &mut src);
  x.eval(t);
}

#[test]
fn test_gpu_io_deserialize_node() {
  println!();
  let x: Val<_> = src(GPUDeviceOuterBatchArray3d::<f32>::zeros_init(([32, 32, 3], 64)));
  let mut src = MemArray4d::<f32>::zeros([32, 32, 3, 64]);
  let t = txn();
  x._to_node().deserialize(t, &mut src);
  x.eval(t);
}

#[test]
fn test_gpu_io_serialize() {
  println!();
  let x = touch(GPUDeviceOuterBatchArray3d::<f32>::uniform_init(([32, 32, 3], 64), -1.0, 1.0, &mut thread_rng()));
  let mut dst = MemArray4d::<f32>::zeros([32, 32, 3, 64]);
  println!("DEBUG: {:?}", &dst.flat_view().unwrap().as_slice()[.. 10]);
  let t = txn();
  x.eval(t);
  x.serialize(t, &mut dst);
  println!("DEBUG: {:?}", &dst.flat_view().unwrap().as_slice()[.. 10]);
}

#[test]
fn test_gpu_io_serialize_node() {
  println!();
  let x = touch(GPUDeviceOuterBatchArray3d::<f32>::uniform_init(([32, 32, 3], 64), -1.0, 1.0, &mut thread_rng()));
  let mut dst = MemArray4d::<f32>::zeros([32, 32, 3, 64]);
  println!("DEBUG: {:?}", &dst.flat_view().unwrap().as_slice()[.. 10]);
  let t = txn();
  x.eval(t);
  x._to_node().serialize(t, &mut dst);
  println!("DEBUG: {:?}", &dst.flat_view().unwrap().as_slice()[.. 10]);
}

#[test]
#[should_panic]
fn test_gpu_mux_fail() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  //let y: Rc<AOp<_>> = x.gpu_mux(GPUDeviceId(1));
  let y: Val<_> = x.gpu_mux(GPUDeviceId(1));
  println!("DEBUG: test: eval...");
  y.eval(txn());
}

#[test]
fn test_gpu_mux() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  //let y: Rc<AOp<_>> = x.gpu_mux(GPUDeviceId(0));
  let y: Val<_> = x.gpu_mux(GPUDeviceId(0));
  println!("DEBUG: test: eval...");
  y.eval(txn());
}

#[test]
fn test_gpu_mux_2() {
  println!();
  let x1 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceOuterBatchArray1d::<f32>::zeros(1024, 1, conn)
  }));
  let x2 = x1.positive_clip();
  let y = x2.gpu_mux(GPUDeviceId(0));
  println!("DEBUG: test: eval...");
  let t = txn();
  y.eval(t);
}

#[test]
fn test_gpu_op_switch() {
  println!();
  //let flag = TCell::new(false);
  let flag = src_init(false);
  let x1 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let x2 = ones(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let y = switch(flag.clone(), x1, x2);
  let t = txn();
  //flag.propose(t, |_| false);
  flag.set(t, false);
  y.eval(t);
  let t = txn();
  //flag.propose(t, |_| true);
  flag.set(t, true);
  y.eval(t);
  // TODO
}

#[test]
fn test_gpu_op_switch_get() {
  println!();
  //let flag = TCell::new(false);
  let flag = src_init(false);
  let x1 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let x2 = ones(Rc::new(|_, conn: GPUDeviceConn| {
    println!("DEBUG: test: allocating...");
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let y = switch(flag.clone(), x1, x2);
  let mut z = MemArray1d::<f32>::zeros(1024);
  let t = txn();
  //flag.propose(t, |_| false);
  flag.set(t, false);
  y.eval(t);
  let _ = y.get(t);
  y.serialize(t, &mut z);
  for k in 0 .. 1024 {
    assert_eq!(z.as_view().as_slice()[k], 0.0);
  }
  println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
  let t = txn();
  //flag.propose(t, |_| true);
  flag.set(t, true);
  y.eval(t);
  let _ = y.get(t);
  y.serialize(t, &mut z);
  for k in 0 .. 1024 {
    assert_eq!(z.as_view().as_slice()[k], 1.0);
  }
  println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
  let t = txn();
  //flag.propose(t, |_| false);
  flag.set(t, false);
  y.eval(t);
  let _ = y.get(t);
  y.serialize(t, &mut z);
  for k in 0 .. 1024 {
    assert_eq!(z.as_view().as_slice()[k], 0.0);
  }
  println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
}

#[test]
fn test_gpu_adj() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  }));
  let mut x_sink = sink(x.clone());
  let dx = x.adjoint(&mut x_sink);
}

#[test]
fn test_gpu_adj_sumjoin() {
  println!();
  println!("DEBUG: enable graph log...");
  enable_static_graph_logging();
  enable_dynamic_graph_logging();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    //GPUDeviceOuterBatchArray1d::<f32>::zeros(1024, 1, conn)
    GPUDeviceScalar::<f32>::zeros((), conn)
  })).named("x");
  //let y = x.clone() + x.clone();
  //let y = sum(vec![x.clone()]).named("y");
  //let y = sum(vec![x.clone(), x.clone()]).named("y");
  let y = sum(vec![
      x.clone(), x.clone(), x.clone(), x.clone(), x.clone(),
      x.clone(), x.clone(), x.clone(), x.clone(), x.clone(),
  ]).named("y");
  let mut y_sink = sink(y.clone());
  let dx = x.adjoint(&mut y_sink).unwrap();
  let t = txn();
  y.eval(t);
  dx.eval(t);
  let mut z: f32 = -1.0;
  dx.serialize(t, &mut z);
  println!("DEBUG: {:?}", z);
  assert_eq!(z, 10.0);
}

#[test]
fn test_gpu_adj_sumjoin2() {
  println!();
  println!("DEBUG: enable graph log...");
  enable_static_graph_logging();
  enable_dynamic_graph_logging();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    //GPUDeviceOuterBatchArray1d::<f32>::zeros(1024, 1, conn)
    GPUDeviceScalar::<f32>::zeros((), conn)
  })).named("x");
  let y = sum_inplace_unstable(vec![
      x.clone(), x.clone(), x.clone(), x.clone(), x.clone(),
      x.clone(), x.clone(), x.clone(), x.clone(), x.clone(),
  ]).named("y");
  let mut y_sink = sink(y.clone());
  let dx = x.adjoint(&mut y_sink).unwrap();
  let t = txn();
  y.eval(t);
  dx.eval(t);
  let mut z: f32 = -1.0;
  dx.serialize(t, &mut z);
  println!("DEBUG: {:?}", z);
  assert_eq!(z, 10.0);
}

#[test]
fn test_gpu_adj_tricky_case() {
  println!();
  println!("DEBUG: enable graph log...");
  enable_static_graph_logging();
  enable_dynamic_graph_logging();
  let target = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchScalar::<f32>::zeros((), 16, conn)
  })).named("target");
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  })).named("x");
  let tmp = x.clone().batch_broadcast(16).named("tmp");
  //let tmp = x.clone().batch_broadcast_like(target.clone().fix()).named("tmp");
  let tmp2 = tmp.clone().batch_sum().named("tmp2");
  let y = sum(vec![x.clone(), tmp2.clone()]).named("y");
  //let y = sum(vec![tmp2.clone(), x.clone()]).named("y");
  let y = y.batch_broadcast(16).batch_sum();
  println!("DEBUG: build adjoint...");
  let mut y_sink = sink(y.clone());
  println!("DEBUG: done building adjoint");
  //dump_static_graph();
  println!("DEBUG: query dy...");
  let dy = y.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dy key: {:?}", dy._graph_key());
  println!("DEBUG: query dt2...");
  let dt2 = tmp2.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dt2 key: {:?}", dt2._graph_key());
  println!("DEBUG: query dt...");
  let dt = tmp.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dt key: {:?}", dt._graph_key());
  println!("DEBUG: query dx...");
  let dx = x.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dx key: {:?}", dx._graph_key());
  println!("DEBUG: done querying");
  let t = txn();
  target.eval(t);
  //tmp.eval(t);
  //tmp2.eval(t);
  //println!("DEBUG: eval y...");
  //y.eval(t);
  //dy.eval(t);
  //println!("DEBUG: eval dt2...");
  //dt2.eval(t);
  // FIXME: why is `dt` not correctly eval'd?
  // FIXME: has to do with the use of inplace sum-join in adjoint building.
  //println!("DEBUG: eval dt...");
  //dt.eval(t);
  println!("DEBUG: eval dx...");
  dx.eval(t);
  let mut z: f32 = -1.0;
  dx.serialize(t, &mut z);
  println!("DEBUG: {:?}", z);
  assert_eq!(z, 272.0);
}

#[test]
fn test_gpu_adj_tricky_case2() {
  println!();
  println!("DEBUG: enable graph log...");
  enable_static_graph_logging();
  enable_dynamic_graph_logging();
  let target = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchScalar::<f32>::zeros((), 16, conn)
  })).named("target");
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  })).named("x");
  let tmp = x.clone().batch_broadcast(16).named("tmp");
  //let tmp = x.clone().batch_broadcast_like(target.clone().fix()).named("tmp");
  let tmp2 = tmp.clone().batch_sum().named("tmp2");
  let y = sum_inplace_unstable(vec![x.clone(), tmp2.clone()]).named("y"); // NOTE: this yields 256 (wrong).
  //let y = sum_inplace_unstable(vec![x.clone().pass(), tmp2.clone()]).named("y"); // NOTE: this yields 272 (expected).
  //let y = sum_inplace_unstable(vec![tmp2.clone(), x.clone()]).named("y"); // NOTE: this also yields 256 (wrong).
  //let y = sum_inplace_unstable(vec![tmp2.clone().pass(), x.clone()]).named("y"); // NOTE: this also yields 256 (wrong).
  let y = y.batch_broadcast(16).batch_sum();
  println!("DEBUG: build adjoint...");
  let mut y_sink = sink(y.clone());
  println!("DEBUG: done building adjoint");
  //dump_static_graph();
  /*println!("DEBUG: query dy...");
  let dy = y.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dy key: {:?}", dy._graph_key());
  println!("DEBUG: query dt2...");
  let dt2 = tmp2.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dt2 key: {:?}", dt2._graph_key());
  println!("DEBUG: query dt...");
  let dt = tmp.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dt key: {:?}", dt._graph_key());*/
  println!("DEBUG: query dx...");
  let dx = x.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dx key: {:?}", dx._graph_key());
  println!("DEBUG: done querying");
  let t = txn();
  target.eval(t);
  //tmp.eval(t);
  //tmp2.eval(t);
  //println!("DEBUG: eval y...");
  //y.eval(t);
  //dy.eval(t);
  //println!("DEBUG: eval dt2...");
  //dt2.eval(t);
  // FIXME: why is `dt` not correctly eval'd?
  // FIXME: has to do with the use of inplace sum-join in adjoint building.
  //println!("DEBUG: eval dt...");
  //dt.eval(t);
  println!("DEBUG: eval dx...");
  dx.eval(t);
  let mut z: f32 = -1.0;
  dx.serialize(t, &mut z);
  println!("DEBUG: {:?}", z);
  assert_eq!(z, 272.0);
}

#[test]
fn test_gpu_adj_tricky_case3() {
  println!();
  println!("DEBUG: enable graph log...");
  enable_static_graph_logging();
  enable_dynamic_graph_logging();
  let target = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchScalar::<f32>::zeros((), 16, conn)
  })).named("target");
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  })).named("x");
  let tmp = x.clone().batch_broadcast(16).named("tmp");
  //let tmp = x.clone().batch_broadcast_like(target.clone().fix()).named("tmp");
  let tmp2 = tmp.clone().batch_sum().named("tmp2");
  //let y = sum_inplace_unstable(vec![x.clone(), tmp2.clone()]).named("y"); // NOTE: this yields 256 (wrong).
  let y = sum_inplace_unstable(vec![x.clone().pass(), tmp2.clone()]).named("y"); // NOTE: this yields 272 (expected).
  //let y = sum_inplace_unstable(vec![tmp2.clone(), x.clone()]).named("y"); // NOTE: this also yields 256 (wrong).
  //let y = sum_inplace_unstable(vec![tmp2.clone().pass(), x.clone()]).named("y"); // NOTE: this also yields 256 (wrong).
  let y = y.batch_broadcast(16).batch_sum();
  println!("DEBUG: build adjoint...");
  let mut y_sink = sink(y.clone());
  println!("DEBUG: done building adjoint");
  println!("DEBUG: query dx...");
  let dx = x.adjoint(&mut y_sink).unwrap();
  println!("DEBUG:   dx key: {:?}", dx._graph_key());
  println!("DEBUG: done querying");
  let t = txn();
  target.eval(t);
  println!("DEBUG: eval dx...");
  dx.eval(t);
  let mut z: f32 = -1.0;
  dx.serialize(t, &mut z);
  println!("DEBUG: {:?}", z);
  assert_eq!(z, 272.0);
}

#[test]
fn test_gpu_op_switch_adj() {
  println!();
  //let flag = TCell::default();
  let flag = src_init(false);
  let x1 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  })).named("x1");
  let x2 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  })).named("x2");
  let y = switch(flag.clone(), x1.clone(), x2.clone()).named("y");
  println!("DEBUG: building adjoint");
  let mut y_sink = sink(y.clone());
  println!("DEBUG: getting adjoints");
  let dy = y.adjoint(&mut y_sink).unwrap();
  let dx1 = x1.adjoint(&mut y_sink).unwrap();
  let dx2 = x2.adjoint(&mut y_sink).unwrap();
  println!("DEBUG: ON path (expect 2 allocs)");
  let t = txn();
  //flag.propose(t, |_| true);
  flag.set(t, true);
  dx1.eval(t);
  dx2.eval(t);
  let _ = dx1.get(t);
  let _ = dx2.get(t);
  println!("DEBUG: OFF path (expect 1 alloc)");
  let t2 = txn();
  //flag.propose(t2, |_| false);
  flag.set(t2, false);
  dx1.eval(t2);
  dx2.eval(t2);
  let _ = dx1.get(t2);
  let _ = dx2.get(t2);
  println!("DEBUG: ON path (expect no allocs)");
  let t3 = txn();
  //flag.propose(t3, |_| true);
  flag.set(t3, true);
  dx1.eval(t3);
  dx2.eval(t3);
  let _ = dx1.get(t3);
  let _ = dx2.get(t3);
}

#[test]
//#[should_panic]
fn test_gpu_adj_1d() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let mut x_sink = sink(x.clone());
  let dx = x.adjoint(&mut x_sink);
}

#[test]
fn test_gpu_op_online_avg() {
  println!();
  let x = ones(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let y = src(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let alpha = src_init(0.0_f32);
  //let y = y.online_average(alpha.clone(), x);
  let y = <OnlineAverageOp as OnlineAverageOpExt<_, _>>::build(alpha.clone(), x, y);
  let t = txn();
  //alpha.propose(t, |_| 0.25);
  alpha.set(t, 0.25);
  y.eval(t);
  let mut z = MemArray1d::<f32>::zeros(1024);
  y.serialize(t, &mut z);
  for k in 0 .. 1024 {
    assert_eq!(z.as_view().as_slice()[k], 0.25);
  }
  println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
  let t = txn();
  //alpha.propose(t, |_| 0.25);
  alpha.set(t, 0.25);
  y.eval(t);
  let mut z = MemArray1d::<f32>::zeros(1024);
  y.serialize(t, &mut z);
  for k in 0 .. 1024 {
    assert_eq!(z.as_view().as_slice()[k], 0.4375);
  }
  println!("DEBUG: {:?}", &z.as_view().as_slice()[.. 10]);
}

#[test]
fn test_gpu_op_softmax() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchArray1d::<f32>::zeros(32, 1, conn)
  }));
  let y = x.softmax();

  let mut z = MemArray2d::<f32>::zeros([32, 1]);

  let t = txn();
  y.eval(t);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[..]);
  y.serialize(t, &mut z);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[..]);
  for k in 0 .. 32 {
    assert_eq!(z.as_view().flat_slice().unwrap()[k], 1.0 / 32.0);
  }
}

#[test]
fn test_gpu_op_softmax_cat_nll() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchArray1d::<f32>::zeros(32, 2, conn)
  }));
  let data = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchScalar::<u32>::zeros((), 2, conn)
  }));
  let (nll, y) = x.softmax_categorical_nll(data.clone());

  let mut target = MemArray1d::<u32>::zeros(2);
  target.as_view_mut().flat_slice_mut().unwrap()
    .copy_from_slice(&[12, 17]);

  let mut z = MemArray2d::<f32>::zeros([32, 2]);
  let mut nll_h = MemArray1d::<f32>::zeros(2);
  let mut data_h = MemArray1d::<u32>::zeros(2);

  let t = txn();
  data.deserialize(t, &mut target);
  nll.eval(t);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &nll_h.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &data_h.as_view().flat_slice().unwrap()[..]);
  y.serialize(t, &mut z);
  nll.serialize(t, &mut nll_h);
  data.serialize(t, &mut data_h);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &nll_h.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &data_h.as_view().flat_slice().unwrap()[..]);
  for k in 0 .. 32 * 2 {
    assert_eq!(z.as_view().flat_slice().unwrap()[k], 1.0 / 32.0);
  }
}

#[test]
fn test_gpu_op_softmax_cat_nll_out_of_bounds_nan() {
  println!();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchArray1d::<f32>::zeros(32, 2, conn)
  }));
  let data = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchScalar::<u32>::zeros((), 2, conn)
  }));
  let (nll, y) = x.softmax_categorical_nll(data.clone());
  let loss = nll.clone().batch_sum();

  let mut target = MemArray1d::<u32>::zeros(2);
  target.as_view_mut().flat_slice_mut().unwrap()
    .copy_from_slice(&[12, 42]);

  let mut z = MemArray2d::<f32>::zeros([32, 2]);
  let mut nll_h = MemArray1d::<f32>::zeros(2);
  let mut data_h = MemArray1d::<u32>::zeros(2);
  let mut loss_h: f32 = 0.0;

  let t = txn();
  data.deserialize(t, &mut target);
  loss.eval(t);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &nll_h.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &data_h.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", loss_h);
  y.serialize(t, &mut z);
  nll.serialize(t, &mut nll_h);
  data.serialize(t, &mut data_h);
  loss.serialize(t, &mut loss_h);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &nll_h.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &data_h.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", loss_h);
  for k in 0 .. 32 * 2 {
    assert_eq!(z.as_view().flat_slice().unwrap()[k], 1.0 / 32.0);
  }
}

#[test]
fn test_gpu_op_softmax_cat_nll_sum_adj() {
  println!();
  println!("DEBUG: enable graph log...");
  enable_static_graph_logging();
  enable_dynamic_graph_logging();
  let x = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchArray1d::<f32>::zeros(32, 2, conn)
  })).named("x");
  let data = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchScalar::<u32>::zeros((), 2, conn)
  })).named("data");
  let (nll, _) = x.clone().softmax_categorical_nll(data.clone());
  let nll = nll.named("nll");
  let loss = nll.clone().batch_sum().named("sum");

  let mut loss_sink = sink(loss.clone());
  let nll_adj = nll.adjoint(&mut loss_sink).unwrap();
  let dx = x.adjoint(&mut loss_sink).unwrap();

  let mut target = MemArray1d::<u32>::zeros(2);
  target.as_view_mut().flat_slice_mut().unwrap()
    .copy_from_slice(&[12, 17]);

  let mut nll_h = MemArray1d::<f32>::zeros(2);
  let mut nll_adj_h = MemArray1d::<f32>::zeros(2);
  let mut z = MemArray2d::<f32>::zeros([32, 2]);
  let mut dz = MemArray2d::<f32>::zeros([32, 2]);
  let mut loss_h: f32 = 0.0;

  let t = txn();
  data.deserialize(t, &mut target);
  loss.eval(t);
  dx.eval(t);
  nll.serialize(t, &mut nll_h);
  nll_adj.serialize(t, &mut nll_adj_h);
  x.serialize(t, &mut z);
  dx.serialize(t, &mut dz);
  loss.serialize(t, &mut loss_h);
  println!("DEBUG: {:?}", &nll_h.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &nll_adj_h.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", &dz.as_view().flat_slice().unwrap()[..]);
  println!("DEBUG: {:?}", loss_h);
}

#[test]
fn test_gpu_op_dequantize() {
  println!();
  let x = ones(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchArray3d::<u8>::zeros([32, 32, 3], 1, conn)
  }));
  let y = x.dequantize(0.0_f32, 1.0_f32);

  let t = txn();
  y.eval(t);

  let mut z = MemArray4d::<f32>::zeros([32, 32, 3, 1]);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[.. 10]);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[32 * 32 * 3 - 10 ..]);
  y.serialize(t, &mut z);
  for k in 0 .. 32 * 32 * 3 {
    assert_eq!(z.as_view().flat_slice().unwrap()[k], 1.0 / 255.0);
  }
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[.. 10]);
  println!("DEBUG: {:?}", &z.as_view().flat_slice().unwrap()[32 * 32 * 3 - 10 ..]);
}

#[test]
fn test_gpu_gradient_descent() {
  println!();
  enable_static_graph_logging();
  enable_dynamic_graph_logging();

  let step_size: Val<f32> = src_init(0.0);
  let momentum: Val<f32> = src_init(0.0);

  let x = src(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceScalar::<f32>::zeros((), conn)
  }));
  let y = x.clone() * x.clone();

  let mut y_sink = sink(y.clone());
  let adj_x = x.adjoint(&mut y_sink).unwrap();

  let grad_step = x.clone().gradient_momentum_step(step_size.clone(), momentum.clone(), adj_x.clone());

  let mut x_h: f32 = 1.0;
  //let mut y_h: f32 = 0.0;
  let mut adj_x_h: f32 = 0.0;

  for iter_nr in 0 .. 10 {
    println!("DEBUG: iteration {}...", iter_nr);

    let t = txn();
    if iter_nr == 0 {
      x.deserialize(t, &mut x_h);
    } else {
      x.persist(t);
    }
    //y.eval(t);
    adj_x.eval(t);

    x.serialize(t, &mut x_h);
    //y.serialize(t, &mut y_h);
    adj_x.serialize(t, &mut adj_x_h);
    println!("DEBUG:   x:  {:?}", x_h);
    //println!("DEBUG: y:  {:?}", y_h);
    println!("DEBUG:   dx: {:?}", adj_x_h);

    let step_t = txn();
    step_size.set(step_t, -0.05);
    momentum.set(step_t, 0.0);
    adj_x.persist(step_t);
    grad_step.eval(step_t);
  }
}

#[test]
fn test_gpu_gradient_adam_step() {
  println!();
  enable_static_graph_logging();
  enable_dynamic_graph_logging();

  let step_size: Val<f32> = src_init(0.0);
  //let momentum: Val<f32> = src_init(0.0);
  let decay_rate_1: Val<f32> = src_init(0.0);
  let decay_rate_2: Val<f32> = src_init(0.0);
  let iter_ct: Val<i32> = src_init(0);
  let epsilon: Val<f32> = src_init(0.0);

  let x = src(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1, conn)
  }));
  let y = x.clone() * x.clone();

  let mut y_sink = sink(y.clone());
  let adj_x = x.adjoint(&mut y_sink).unwrap();

  let grad_step = x.clone().adam_step(step_size.clone(), decay_rate_1.clone(), decay_rate_2.clone(), iter_ct.clone(), epsilon.clone(), adj_x.clone());

  let mut x_h: MemArray1d<f32> = MemArray::zeros(1);
  x_h.as_view_mut().flat_slice_mut().unwrap()[0] = 1.0;
  //let mut y_h: f32 = 0.0;
  let mut adj_x_h: MemArray1d<f32> = MemArray::zeros(1);

  for iter_nr in 0 .. 10 {
    println!("DEBUG: iteration {}...", iter_nr);

    let t = txn();
    if iter_nr == 0 {
      x.deserialize(t, &mut x_h);
    } else {
      x.persist(t);
    }
    y.eval(t);
    adj_x.eval(t);

    x.serialize(t, &mut x_h);
    //y.serialize(t, &mut y_h);
    adj_x.serialize(t, &mut adj_x_h);
    println!("DEBUG:   x:  {:?}", x_h.as_view().flat_slice().unwrap()[0]);
    //println!("DEBUG: y:  {:?}", y_h);
    println!("DEBUG:   dx: {:?}", adj_x_h.as_view().flat_slice().unwrap()[0]);

    let step_t = txn();
    step_size.set(step_t, -0.05);
    decay_rate_1.set(step_t, 0.1);
    decay_rate_2.set(step_t, 0.001);
    iter_ct.set(step_t, (iter_nr + 1) as i32);
    epsilon.set(step_t, 1.0e-8);
    adj_x.persist(step_t);
    grad_step.eval(step_t);
  }
}

#[test]
fn test_gpu_op_io_vectorize() {
  println!();
  let x1 = ones(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let x2 = ones(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let mut nodes = NodeVec::new();
  nodes.push_val(x1.clone());
  nodes.push_val(x2.clone());
  let y: Val<GPUDeviceArray1d<f32>> = nodes.vectorize();
  let t = txn();
  y.eval(t);
  let mut y_h = MemArray1d::<f32>::zeros(2048);
  y.serialize(t, &mut y_h);
  println!("DEBUG: {:?} {:?}", &y_h.as_view().flat_slice().unwrap()[ .. 10], &y_h.as_view().flat_slice().unwrap()[2048 - 10 .. ]);
  for k in 0 .. 2048 {
    assert_eq!(y_h.as_view().flat_slice().unwrap()[k], 1.0);
  }
}

#[test]
fn test_gpu_op_io_devectorize() {
  println!();
  let x1 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let x2 = zeros(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(1024, conn)
  }));
  let y = ones(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceArray1d::<f32>::zeros(2048, conn)
  }));
  let mut nodes = NodeVec::new();
  nodes.push_val(x1.clone());
  nodes.push_val(x2.clone());
  let op = y.devectorize(nodes);
  let t = txn();
  op.eval(t);
  let mut x1_h = MemArray1d::<f32>::zeros(1024);
  let mut x2_h = MemArray1d::<f32>::zeros(1024);
  x1.serialize(t, &mut x1_h);
  x2.serialize(t, &mut x2_h);
  println!("DEBUG: {:?} {:?}", &x1_h.as_view().flat_slice().unwrap()[ .. 10], &x1_h.as_view().flat_slice().unwrap()[1024 - 10 .. ]);
  println!("DEBUG: {:?} {:?}", &x2_h.as_view().flat_slice().unwrap()[ .. 10], &x2_h.as_view().flat_slice().unwrap()[1024 - 10 .. ]);
  for k in 0 .. 1024 {
    assert_eq!(x1_h.as_view().flat_slice().unwrap()[k], 1.0);
    assert_eq!(x2_h.as_view().flat_slice().unwrap()[k], 1.0);
  }
}

#[test]
fn test_gpu_op_conv3d() {
  println!();
  let x = ones(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchArray4d::<f32>::zeros([6, 7, 8, 9], 1, conn)
  }));
  let w = src(GPUDeviceArray5d::<f32>::ones_init([3, 3, 3, 9, 5]));
  let mut conv1 = Conv3dShape::default_ncdhw();
  conv1.src_dims = [6, 7, 8];
  conv1.src_features = 9;
  conv1.ker_dims = [3, 3, 3];
  conv1.features = 5;
  conv1.stride = [1, 1, 1];
  conv1.zero_pad = [0, 0, 0];
  let y = w.clone().conv(conv1, x.clone());
  let mut y_h = MemArray5d::<f32>::zeros([4, 5, 6, 5, 1]);
  let y_len = 4 * 5 * 6 * 5 * 1;

  let t = txn();
  w.persist(t);
  y.eval(t);
  y.serialize(t, &mut y_h);
  println!("DEBUG: y: {:?}", &y_h.as_view().flat_slice().unwrap()[ .. 10]);
  println!("DEBUG: y: {:?}", &y_h.as_view().flat_slice().unwrap()[y_len - 10 .. ]);
  for k in 0 .. y_len {
    assert_eq!(y_h.as_view().flat_slice().unwrap()[k], 243.0);
  }
}

#[test]
fn test_gpu_op_conv3d_big() {
  println!();
  let x = ones(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchArray4d::<f32>::zeros([100, 100, 100, 9], 1, conn)
  }));
  let w = src(GPUDeviceArray5d::<f32>::ones_init([1, 1, 1, 9, 5]));
  let mut conv1 = Conv3dShape::default_ncdhw();
  conv1.src_dims = [100, 100, 100];
  conv1.src_features = 9;
  conv1.ker_dims = [1, 1, 1];
  conv1.features = 5;
  conv1.stride = [1, 1, 1];
  conv1.zero_pad = [0, 0, 0];
  let y = w.clone().conv(conv1, x.clone());
  let scale = src_init(0.0_f32);
  let y = y * scale.clone();
  let mut y_h = MemArray5d::<f32>::zeros([100, 100, 100, 5, 1]);
  let y_len = 100 * 100 * 100 * 5 * 1;

  let t = txn();
  w.persist(t);
  scale.set(t, 1.0e-6);
  y.eval(t);
  y.serialize(t, &mut y_h);
  println!("DEBUG: y: {:?}", &y_h.as_view().flat_slice().unwrap()[ .. 10]);
  println!("DEBUG: y: {:?}", &y_h.as_view().flat_slice().unwrap()[y_len - 10 .. ]);
  for k in 0 .. y_len {
    assert_eq!(y_h.as_view().flat_slice().unwrap()[k], 9.0 * 1.0e-6);
  }
}

#[test]
fn test_gpu_op_transpose_conv3d() {
  println!();
  let x = ones(Rc::new(|_, conn: GPUDeviceConn| {
    GPUDeviceOuterBatchArray4d::<f32>::zeros([3, 4, 5, 9], 1, conn)
  }));
  let w = src(GPUDeviceArray5d::<f32>::ones_init([2, 2, 2, 9, 5]));
  let mut conv1 = Conv3dShape::default_ncdhw();
  conv1.src_dims = [6, 8, 10];
  conv1.src_features = 5;
  conv1.ker_dims = [2, 2, 2];
  conv1.features = 9;
  conv1.stride = [2, 2, 2];
  conv1.zero_pad = [0, 0, 0];
  let y = w.clone().left_transpose_conv(conv1, x.clone());
  let mut y_h = MemArray5d::<f32>::zeros([6, 8, 10, 5, 1]);
  let y_len = 6 * 8 * 10 * 5 * 1;

  let t = txn();
  w.persist(t);
  y.eval(t);
  y.serialize(t, &mut y_h);
  println!("DEBUG: y: {:?}", &y_h.as_view().flat_slice().unwrap()[ .. 10]);
  println!("DEBUG: y: {:?}", &y_h.as_view().flat_slice().unwrap()[y_len - 10 .. ]);
  for k in 0 .. y_len {
    assert_eq!(y_h.as_view().flat_slice().unwrap()[k], 9.0);
  }
}
