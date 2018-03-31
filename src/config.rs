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

use std::env;

pub struct Config {
  pub deterministic:    bool,
}

impl Default for Config {
  fn default() -> Self {
    Config{
      deterministic:
        env::var("ANODE_CFG_DETERMINISTIC").ok()
          .and_then(|val| match val.parse() {
            Err(_) => { println!("WARNING: config: failed to parse key 'ANODE_CFG_DETERMINISTIC'"); None }
            Ok(x) => Some(x),
          })
          .unwrap_or(false),
    }
  }
}
