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
