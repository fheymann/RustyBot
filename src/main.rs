use ndarray::prelude::*;
use ndarray::{Array, Ix2};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::Value;
use std::io::{self, BufRead, Write};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Highlight {
    highlight: Vec<(usize, usize, String)>,
}

impl Highlight {
    pub fn empty() -> Self {
        Self {
            highlight: Vec::new(),
        }
    }
    pub fn color(mut self, pos_vec: Vec<Pos>, color_str: &str, mut alpha: u8) -> Self {
        if alpha > 100 {
            alpha = 100;
        }
        for pos in pos_vec {
            self.highlight
                .push((pos.x, pos.y, format! {"{color_str}{alpha}"}.to_string()))
        }
        self
    }

    pub fn blue(mut self, pos_vec: Vec<Pos>, alpha: u8) -> Self {
        self.color(pos_vec, "#0000ff", alpha)
    }

    pub fn white(mut self, pos_vec: Vec<Pos>, alpha: u8) -> Self {
        self.color(pos_vec, "#ffffff", alpha)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Pos {
    x: usize,
    y: usize,
}

impl TryFrom<&Value> for Pos {
    type Error = String;
    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        if let Value::Array(xy_array) = value {
            if let (Some(Value::Number(x_num)), Some(Value::Number(y_num))) =
                (xy_array.get(0), xy_array.get(1))
            {
                Ok(Pos {
                    x: x_num.as_u64().unwrap().try_into().unwrap(),
                    y: y_num.as_u64().unwrap().try_into().unwrap(),
                })
            } else {
                Err("Unable to parse xy Pos: {value:?}".to_string())
            }
        } else {
            Err("Unable to parse xy Pos: {value:?}".to_string())
        }
    }
}

pub enum PixelType {
    Floor,
    Wall,
}
impl PixelType {
    fn as_u8(&self) -> u8 {
        match self {
            PixelType::Floor => 0,
            PixelType::Wall => 1,
        }
    }
}

impl From<PixelType> for u8 {
    fn from(value: PixelType) -> Self {
        value.as_u8()
    }
}

impl TryFrom<u8> for PixelType {
    type Error = u8;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Floor),
            1 => Ok(Self::Wall),
            _ => Err(value),
        }
    }
}

pub fn map_to_pos_vec(ref_map: &PixelMap, pixel_type: PixelType) -> Vec<Pos> {
    let mut ret_vec = Vec::new();
    for ((x, y), v) in ref_map.indexed_iter() {
        if *v == pixel_type.as_u8() {
            ret_vec.push(Pos { x, y })
        }
    }
    ret_vec
}

pub type PixelMap = Array<u8, Ix2>;

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    let moves = ["N", "S", "E", "W", "WAIT"];
    let mut first_tick = true;
    let mut opt_map: Option<PixelMap> = None;
    // Read one JSON object per line from stdin
    for line in io::stdin().lock().lines() {
        if line.is_err() {
            break;
        }
        let line = line.unwrap();

        let data = if !line.trim().is_empty() {
            serde_json::from_str::<Value>(&line).ok()
        } else {
            None
        };

        if first_tick {
            if let Some(Value::Object(map)) = data.as_ref() {
                if let Some(Value::Object(cfg_map)) = map.get("config") {
                    let width = cfg_map
                        .get("width")
                        .map(|v| v.as_u64().unwrap_or(0))
                        .unwrap_or(0);
                    let height = cfg_map
                        .get("height")
                        .map(|v| v.as_u64().unwrap_or(0))
                        .unwrap_or(0);
                    //eprintln!("Random walker (Rust) launching on a {width}x{height} map");
                    opt_map = Some(Array::from_elem(
                        (width.try_into().unwrap(), height.try_into().unwrap()).f(),
                        0 as u8,
                    ));
                    first_tick = false;
                }
            }
        }

        if let Some(Value::Object(map)) = data.as_ref() {
            if let Some(Value::Array(bot_pos)) = map.get("bot") {
                //eprintln!("Map: {rust_map:?}");
                eprintln!("Bot position: {bot_pos:?}");
            }
            if let Some(Value::Array(wall_pixels)) = map.get("wall") {
                for wall_pixel in wall_pixels {
                    let wall_pos: Pos = wall_pixel.try_into().unwrap();
                    if let Some(ref mut tmp_map) = opt_map {
                        //tmp_map[(wall_pixel, 0)] = 17;
                        tmp_map[(wall_pos.x, wall_pos.y)] = PixelType::Wall.into();
                    }
                }
            }
        }

        // Emit a random move
        let move_index = rng.random_range(0..moves.len());
        // Write and flush promptly
        //let highlight_json = "{\"highlight\":[[2,2,\"#00ff0050\"]]}";
        //let pos_vec = vec![Pos { x: 2, y: 2 }];
        let wall_pos_vec = map_to_pos_vec(opt_map.as_ref().unwrap(), PixelType::Wall);
        let highlight = Highlight::empty().blue(wall_pos_vec, 70);
        let highlight_json = serde_json::to_string(&highlight).unwrap();
        //eprintln!("{highlight_json}");
        println!("{} {highlight_json}", moves[move_index]);
        let _ = io::stdout().flush();
    }
}
