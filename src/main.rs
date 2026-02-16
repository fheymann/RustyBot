use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::Value;
use std::io::{self, BufRead, Write};

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    let moves = ["N", "S", "E", "W"];
    let mut first_tick = true;

    // Read one JSON object per line from stdin
    for line in io::stdin().lock().lines() {
        if line.is_err() {
            break;
        }
        let line = line.unwrap();

        let data = if !line.trim().is_empty() {
            serde_json::from_str::<Value>(&line).map(|v| v).ok()
        } else {
            None
        };

        if first_tick {
            if let Some(Value::Object(map)) = data.as_ref() {
                if let Some(Value::Object(cfg_map)) = map.get("config") {
                    let width = cfg_map.get("width").map(|v| v.as_i64().unwrap_or(0)).unwrap_or(0);
                    let height = cfg_map.get("height").map(|v| v.as_i64().unwrap_or(0)).unwrap_or(0);
                    eprintln!("Random walker (Rust) launching on a {width}x{height} map");
                    first_tick = false;
                }
            }
        }

        // Emit a random move
        let move_index = rng.random_range(0..moves.len());
        // Write and flush promptly
        println!("{}", moves[move_index]);
        let _ = io::stdout().flush();
    }
}

