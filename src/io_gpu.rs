/*
Copyright 2017-2018 Peter Jin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use ::*;
use context::*;
//use ffi::routines_gpu::*;
//use ops::*;

//use arithmetic::*;
//use arrayidx::*;
//use cuda_blas::*;
//use cuda_dnn::*;
use gpudevicemem::*;
use gpudevicemem::array::*;
//use gpudevicemem::array::linalg::*;
use memarray::*;
//use rand::{Rng};

//use std::cell::{RefMut};
//use std::marker::{PhantomData};
//use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
//use std::ops::{Add, Mul};
//use std::sync::{Arc};

impl<T> FlatIO<MemArray1d<T>> where T: Copy {
  pub fn len(&self) -> usize {
    self.buffer.size()
  }

  pub fn next_slice(&mut self, size: usize) -> MemArrayView1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    assert!(next_offset <= self.buffer.size());
    let slice = self.buffer.as_view().view(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }

  pub fn next_slice_mut(&mut self, size: usize) -> MemArrayViewMut1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    assert!(next_offset <= self.buffer.size());
    let slice = self.buffer.as_view_mut().view_mut(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }
}

impl<T> ArrayIO<MemArray1d<T>> where T: Copy {
  pub fn size(&self) -> usize {
    self.array.size()
  }

  pub fn next_view(&mut self, size: usize) -> MemArrayView1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    let slice = self.array.as_view().view(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }

  pub fn next_view_mut(&mut self, size: usize) -> MemArrayViewMut1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    let slice = self.array.as_view_mut().view_mut(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }
}

impl<T> ArrayIO<MemArray2d<T>> where T: Copy {
  pub fn size(&self) -> [usize; 2] {
    self.array.size()
  }

  pub fn next_view(&mut self, size: [usize; 2]) -> MemArrayView2d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view().view(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
    );
    self.offset = next_offset;
    slice
  }

  pub fn next_view_mut(&mut self, size: [usize; 2]) -> MemArrayViewMut2d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view_mut().view_mut(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
    );
    self.offset = next_offset;
    slice
  }
}

impl<T> ArrayIO<MemArray3d<T>> where T: Copy {
  pub fn size(&self) -> [usize; 3] {
    self.array.size()
  }

  pub fn next_view(&mut self, size: [usize; 3]) -> MemArrayView3d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view().view(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
    );
    self.offset = next_offset;
    slice
  }

  pub fn next_view_mut(&mut self, size: [usize; 3]) -> MemArrayViewMut3d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view_mut().view_mut(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
    );
    self.offset = next_offset;
    slice
  }
}

impl<T> ArrayIO<MemArray4d<T>> where T: Copy {
  pub fn size(&self) -> [usize; 4] {
    self.array.size()
  }

  pub fn next_view(&mut self, size: [usize; 4]) -> MemArrayView4d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view().view(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
        prev_offset[3] .. next_offset[3],
    );
    self.offset = next_offset;
    slice
  }

  pub fn next_view_mut(&mut self, size: [usize; 4]) -> MemArrayViewMut4d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let slice = self.array.as_view_mut().view_mut(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
        prev_offset[3] .. next_offset[3],
    );
    self.offset = next_offset;
    slice
  }
}

/*impl<T> BatchArrayIO<OuterBatchMemArray3d<T>> where T: Copy {
  pub fn next_view(&mut self, size: [usize; 3], batch_sz: usize) -> MemArrayView4d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let prev_eloffset = self.eloffset;
    let next_eloffset = self.eloffset + batch_sz;
    let slice = self.array.as_view().view(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
        prev_eloffset .. next_eloffset,
    );
    self.offset = next_offset;
    self.eloffset = next_eloffset;
    slice
  }

  pub fn next_view_mut(&mut self, size: [usize; 3], batch_sz: usize) -> MemArrayViewMut4d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset.index_add(&size);
    let prev_eloffset = self.eloffset;
    let next_eloffset = self.eloffset + batch_sz;
    let slice = self.array.as_view_mut().view_mut(
        prev_offset[0] .. next_offset[0],
        prev_offset[1] .. next_offset[1],
        prev_offset[2] .. next_offset[2],
        prev_eloffset .. next_eloffset,
    );
    self.offset = next_offset;
    self.eloffset = next_eloffset;
    slice
  }
}*/

impl<T> FlatIO<GPUDeviceArray1d<T>> where T: Copy {
  pub fn len(&self) -> usize {
    self.buffer.size()
  }

  pub fn next_slice(&mut self, size: usize) -> GPUDeviceArrayView1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    assert!(next_offset <= self.buffer.size());
    let slice = self.buffer.as_view().view(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }

  pub fn next_slice_mut(&mut self, size: usize) -> GPUDeviceArrayViewMut1d<T> {
    let prev_offset = self.offset;
    let next_offset = self.offset + size;
    assert!(next_offset <= self.buffer.size());
    let slice = self.buffer.as_view_mut().view_mut(prev_offset .. next_offset);
    self.offset = next_offset;
    slice
  }
}

impl<T> IOVal for RWVal<GPUDeviceArray1d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<FlatIO<()>>() {
      // TODO
      unimplemented!();
    }
    if let Some(dst) = dst.downcast_mut::<FlatIO<MemArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.size());
      x.as_view().sync_dump_mem(dst_slice, conn);
      return;
    }
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_view = dst.next_view_mut(x.size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    if let Some(dst) = dst.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.size());
      guard._wait(dst_slice.async_state());
      dst_slice.copy(x.as_view(), conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<FlatIO<()>>() {
      // TODO
      unimplemented!();
    }
    if let Some(src) = src.downcast_mut::<FlatIO<MemArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.size());
            x.as_view_mut().sync_copy_mem(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_view = src.next_view(x.size());
            x.as_view_mut().sync_copy_mem(src_view, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    if let Some(src) = src.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.size());
            guard._wait(src_slice.async_state());
            x.as_view_mut().copy(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceArray2d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray2d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_view = dst.next_view_mut(x.size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    if let Some(dst) = dst.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.flat_size());
      guard._wait(dst_slice.async_state());
      dst_slice.copy(x.flat_view().unwrap(), conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray2d<T>>>() {
      // TODO
      unimplemented!();
    }
    if let Some(src) = src.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.flat_size());
            guard._wait(src_slice.async_state());
            x.flat_view_mut().unwrap().copy(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceArray3d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray3d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_view = dst.next_view_mut(x.size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    if let Some(dst) = dst.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.flat_size());
      guard._wait(dst_slice.async_state());
      dst_slice.copy(x.flat_view().unwrap(), conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray3d<T>>>() {
      // TODO
      unimplemented!();
    }
    if let Some(src) = src.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.flat_size());
            guard._wait(src_slice.async_state());
            x.flat_view_mut().unwrap().copy(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceArray4d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.async_state());
      let mut dst_slice = dst.next_slice_mut(x.flat_size());
      guard._wait(dst_slice.async_state());
      dst_slice.copy(x.flat_view().unwrap(), conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<FlatIO<GPUDeviceArray1d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.async_state());
            let src_slice = src.next_slice(x.flat_size());
            guard._wait(src_slice.async_state());
            x.flat_view_mut().unwrap().copy(src_slice, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceOuterBatchArray1d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray2d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.as_view().async_state());
      let mut dst_view = dst.next_view_mut(x.as_view().size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray2d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.as_view().async_state());
            x.set_batch_size(src.size()[1]);
            let src_view = src.next_view(x.as_view().size());
            x.as_view_mut().sync_copy_mem(src_view, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}

impl<T> IOVal for RWVal<GPUDeviceOuterBatchArray3d<T>> where T: Copy + 'static {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    if let Some(dst) = dst.downcast_mut::<ArrayIO<MemArray4d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      let mut section = GPULazyAsyncSection::default();
      let mut guard = section.enter(conn.clone());
      let x = self.get(txn, rvar);
      guard._wait(x.as_view().async_state());
      let mut dst_view = dst.next_view_mut(x.as_view().size());
      x.as_view().sync_dump_mem(dst_view, conn);
      return;
    }
    unimplemented!();
  }

  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    println!("DEBUG: GPUDeviceOuterBatchArray3d<T>::_deserialize");
    /*if let Some(src) = src.downcast_mut::<ArrayIO<MemArray3d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.as_view().async_state());
            x.set_batch_size(1);
            // TODO: either upgrade the src view or downgrade the dst view.
            /*let src_view = src.next_view(x.as_view().size());
            x.as_view_mut().sync_copy_mem(src_view, conn);*/
            unimplemented!();
          }
          _ => unimplemented!(),
        }
      }
      return;
    }*/
    if let Some(src) = src.downcast_mut::<ArrayIO<MemArray4d<T>>>() {
      let ctx = implicit_ctx().gpu();
      let mut pool = ctx.pool();
      let conn = pool.conn();
      if let Some((cap, token)) = self.write(txn, xvar) {
        let mut section = GPULazyAsyncSection::default();
        let mut guard = section.enter(conn.clone());
        match cap {
          WriteCap::Assign => {
            let mut x = self.get_mut(txn, xvar, token);
            guard._wait(x.as_view().async_state());
            x.set_batch_size(src.size()[3]);
            let src_view = src.next_view(x.as_view().size());
            x.as_view_mut().sync_copy_mem(src_view, conn);
          }
          _ => unimplemented!(),
        }
      }
      return;
    }
    unimplemented!();
  }
}
