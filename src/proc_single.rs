/*
Copyright 2018 Peter Jin

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

pub struct DistProcGroup {
  closed:   bool,
}

impl Default for DistProcGroup {
  fn default() -> Self {
    DistProcGroup{
      closed:   false,
    }
  }
}

impl Iterator for DistProcGroup {
  type Item = DistProc;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    self.closed = true;
    Some(DistProc{})
  }
}

pub struct DistProcJoinHandle {
}

impl Drop for DistProcJoinHandle {
  fn drop(&mut self) {
    // TODO
  }
}

impl DistProcJoinHandle {
  pub fn join(self) {
  }
}

#[derive(Clone)]
pub struct DistProc {
}

impl DistProc {
  pub fn spawn<F>(self, f: F) -> Result<DistProcJoinHandle, ()> where F: FnOnce(DistProc) + Send + 'static {
    f(self);
    Ok(DistProcJoinHandle{})
  }

  pub fn rank(&self) -> usize {
    0
  }

  pub fn num_ranks(&self) -> usize {
    1
  }

  pub fn barrier(&self) {
  }

  pub fn allreduce_sum<T: Copy>(&self, _buf: &mut [T]) {
  }
}
