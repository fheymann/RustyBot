use ndarray::prelude::*;
use ndarray::{Array, Ix2};
use pathfinding::matrix::directions::W;
use pathfinding::num_traits::sign;
use pathfinding::prelude::astar;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::Value;
use std::collections::VecDeque;
use std::default;
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

    pub fn blue(self, pos_vec: Vec<Pos>, alpha: u8) -> Self {
        self.color(pos_vec, "#0000ff", alpha)
    }

    pub fn white(self, pos_vec: Vec<Pos>, alpha: u8) -> Self {
        self.color(pos_vec, "#ffffff", alpha)
    }
}

#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct Pos {
    x: usize,
    y: usize,
}

impl Pos {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, ref_pos: &Pos) -> u32 {
        (self.x.abs_diff(ref_pos.x) + self.y.abs_diff(ref_pos.y)) as u32
    }

    pub fn a_star_succesors(&self, ref_pixel_map: &PixelMap) -> Vec<(Pos, u32)> {
        let mut ret_vec = Vec::new();
        if self.x > 0 {
            if ref_pixel_map[(self.x - 1, self.y)] != PixelType::Wall.as_u8() {
                ret_vec.push((Pos::new(self.x - 1, self.y), 1));
            }
        }
        if self.x < ref_pixel_map.dim().0 - 1 {
            if ref_pixel_map[(self.x + 1, self.y)] != PixelType::Wall.as_u8() {
                ret_vec.push((Pos::new(self.x + 1, self.y), 1));
            }
        }
        if self.y > 0 {
            if ref_pixel_map[(self.x, self.y - 1)] != PixelType::Wall.as_u8() {
                ret_vec.push((Pos::new(self.x, self.y - 1), 1));
            }
        }
        if self.y < ref_pixel_map.dim().1 - 1 {
            if ref_pixel_map[(self.x, self.y + 1)] != PixelType::Wall.as_u8() {
                ret_vec.push((Pos::new(self.x, self.y + 1), 1));
            }
        }
        ret_vec
    }
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
    Unknown,
}
impl PixelType {
    fn as_u8(&self) -> u8 {
        match self {
            PixelType::Floor => 0,
            PixelType::Wall => 1,
            PixelType::Unknown => 255,
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
            255 => Ok(Self::Unknown),
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

#[derive(Debug)]
pub struct GemList {
    gem_vec: Vec<Gem>,
}

impl GemList {
    pub fn new() -> Self {
        Self {
            gem_vec: Vec::new(),
        }
    }
    pub fn exists_pos(&self, ref_pos: &Pos) -> bool {
        let mut exists = false;
        for gem in &self.gem_vec {
            if *gem.ref_pos() == *ref_pos {
                exists = true;
                break;
            }
        }
        exists
    }

    pub fn exists_channel(&self, channel_id: usize) -> bool {
        let mut exists: bool = false;
        for gem in &self.gem_vec {
            if gem.channel_id() == channel_id {
                exists = true;
                break;
            }
        }
        exists
    }

    pub fn remove_channel(&mut self, channel_id: usize) {
        let mut opt_remove_idx = None;
        for (idx, gem) in &mut self.gem_vec.iter().enumerate() {
            if gem.channel_id() == channel_id {
                opt_remove_idx = Some(idx);
                break;
            }
        }

        if let Some(remove_idx) = opt_remove_idx {
            self.gem_vec.remove(remove_idx);
        }
    }

    pub fn ref_mut_gem_by_channel(&mut self, channel_id: usize) -> Option<&mut Gem> {
        let mut ref_gem = None;
        for gem in &mut self.gem_vec {
            if gem.channel_id() == channel_id {
                ref_gem = Some(gem);
                break;
            }
        }
        ref_gem
    }

    pub fn add_channel_measurement(&mut self, channel_id: usize, bot_pos: Pos, signal: f64) {
        if let Some(ref mut ref_gem) = self.ref_mut_gem_by_channel(channel_id) {
            eprintln!("add_measurement");
            ref_gem.add_measurement(bot_pos, signal);
        }
    }

    pub fn add_gem_with_pos(&mut self, pos: Pos, ttl: u64) {
        if !self.exists_pos(&pos) {
            self.gem_vec.push(Gem::new_with_pos(pos, ttl));
        } else {
            //eprintln!("Gem exists at pos: {pos:?} with ttl: {ttl}");
        }
    }

    pub fn add_gem_with_channel(
        &mut self,
        ref_pixel_map: &PixelMap,
        bot_pos: Pos,
        ttl: u64,
        channel_id: usize,
    ) {
        if !self.exists_channel(channel_id) {
            self.gem_vec.push(Gem::new_with_channel(
                ref_pixel_map,
                bot_pos,
                ttl,
                channel_id,
            ));
        }
    }

    pub fn remove_gem(&mut self, ref_pos: &Pos) {
        let mut opt_remove_idx = None;
        for (idx, gem) in &mut self.gem_vec.iter().enumerate() {
            if *gem.ref_pos() == *ref_pos {
                opt_remove_idx = Some(idx);
                break;
            }
        }

        if let Some(remove_idx) = opt_remove_idx {
            self.gem_vec.remove(remove_idx);
        }
    }

    pub fn check_bot_pos(&mut self, ref_bot: &Bot) {
        let bot_pos = ref_bot.ref_current_pos().unwrap();
        if self.exists_pos(bot_pos) {
            self.remove_gem(bot_pos);
        }
    }

    pub fn next_tick(&mut self, ref_bot: &Bot) {
        self.check_bot_pos(ref_bot);
        for gem in &mut self.gem_vec {
            gem.next_tick(*ref_bot.ref_current_pos().unwrap());
        }
    }

    pub fn first_known(&self) -> Option<Gem> {
        if !self.known_vec().is_empty() {
            Some(self.known_vec()[0].clone())
        } else {
            None
        }
    }

    pub fn known_pos_vec(&self) -> Vec<Pos> {
        let mut pos_vec = Vec::new();
        for gem in self.known_vec() {
            pos_vec.push(*gem.ref_known_pos().unwrap());
        }
        pos_vec
    }

    pub fn guess_pos_vec(&self) -> Vec<Pos> {
        let mut pos_vec = Vec::new();
        for gem in self.guess_vec() {
            pos_vec.push(*gem.ref_pos());
        }
        pos_vec
    }

    pub fn guess_vec(&self) -> Vec<&Gem> {
        let mut guess_vec = Vec::new();
        for tmp_gem in &self.gem_vec {
            if tmp_gem.ref_known_pos().is_none() {
                guess_vec.push(tmp_gem);
            }
        }
        guess_vec
    }

    pub fn known_vec(&self) -> Vec<&Gem> {
        let mut known_vec = Vec::new();
        for tmp_gem in &self.gem_vec {
            if tmp_gem.ref_known_pos().is_some() {
                known_vec.push(tmp_gem);
            }
        }
        known_vec
    }

    pub fn len(&self) -> usize {
        self.gem_vec.len()
    }
}

#[derive(Debug, Clone)]
pub struct Gem {
    known_pos: Option<Pos>,
    guess_pos: Pos,
    ttl: u64,
    meas_hist: Vec<(Pos, f64, f64)>,
    channel_id: usize,
}

impl Gem {
    pub fn new_with_channel(
        ref_pixel_map: &PixelMap,
        bot_pos: Pos,
        ttl: u64,
        channel_id: usize,
    ) -> Self {
        let guess_pos = Pos::new(ref_pixel_map.dim().0 / 2, ref_pixel_map.dim().1 / 2);
        Self {
            known_pos: None,
            ttl,
            guess_pos,
            channel_id,
            meas_hist: Vec::new(),
        }
    }

    pub fn new_with_pos(pos: Pos, ttl: u64) -> Self {
        Self {
            known_pos: Some(pos),
            ttl,
            guess_pos: pos,
            channel_id: usize::MAX,
            meas_hist: Vec::new(),
        }
    }

    pub fn ref_known_pos(&self) -> Option<&Pos> {
        self.known_pos.as_ref()
    }

    pub fn ref_pos(&self) -> &Pos {
        if let Some(pos) = self.known_pos.as_ref() {
            pos
        } else {
            &self.guess_pos
        }
    }

    pub fn channel_id(&self) -> usize {
        self.channel_id
    }

    pub fn update_ttl(&mut self, ttl: u64) {
        self.ttl = ttl;
    }

    pub fn ttl(&self) -> u64 {
        self.ttl
    }

    pub fn add_measurement(&mut self, bot_pos: Pos, signal: f64) {
        let signal_fade = 10.;
        let signal_radius = 7.;
        let tmp_fade = (301. - self.ttl as f64) / signal_fade;
        let fade = if tmp_fade < 1. { tmp_fade } else { 1. };
        eprintln!("fade: {fade}");
        self.meas_hist.push((bot_pos, fade, signal));
    }

    pub fn next_tick(&mut self, bot_pos: Pos) {
        self.ttl -= 1;
        if self.known_pos.is_none() {
            let current_guess = self.guess_pos;
            let mut find_min = 1e99;
            let mut new_guess_x = current_guess.x;
            let mut new_guess_y = current_guess.y;
            for _ in 0..10 {
                for ix in vec![-1i64, 0i64, 1i64] {
                    for iy in vec![-1i64, 0i64, 1i64] {
                        let new_x = current_guess.x as i64 + ix;
                        let new_y = current_guess.y as i64 + iy;
                        let mut tmp_min = 0.;
                        for (meas_pos, fade, meas_signal) in &self.meas_hist {
                            let delta_x = new_x - meas_pos.x as i64;
                            let delta_y = new_y - meas_pos.y as i64;
                            let distance =
                                ((delta_x as f64).powf(2.) + (delta_y as f64).powf(2.)).sqrt();
                            let signal_radius = 7.;
                            let gem_signal = fade / (1. + (distance / signal_radius).powf(2.0));
                            tmp_min += (gem_signal - meas_signal).powf(2.);
                        }
                        if tmp_min < find_min {
                            find_min = tmp_min;
                            new_guess_x = new_x as usize;
                            new_guess_y = new_y as usize;
                        }
                    }
                }
            }
            self.guess_pos = Pos::new(new_guess_x, new_guess_y);
            eprintln!("guessing new Pos: {:?})", self.guess_pos);
        }
    }
}

pub type PixelMap = Array<u8, Ix2>;

pub fn find_unknown_pos(ref_pixel_map: &PixelMap) -> Pos {
    let mut pos_x = 0;
    let mut pos_y = 0;
    for ((x, y), v) in ref_pixel_map.indexed_iter() {
        if x > 0
            && x < ref_pixel_map.dim().0 - 1
            && y > 0
            && y < ref_pixel_map.dim().1 - 1
            && *v == PixelType::Unknown.as_u8()
        {
            pos_x = x;
            pos_y = y;
            break;
        }
    }
    Pos::new(pos_x, pos_y)
}

pub struct Bot {
    pos_hist: VecDeque<Pos>,
    //current_path: VecDeque<Pos>,
}

impl Bot {
    pub fn new() -> Self {
        Self {
            pos_hist: VecDeque::new(),
            //current_path: VecDeque::new(),
        }
    }

    pub fn update_pos(&mut self, pos: Pos) {
        self.pos_hist.push_back(pos);
    }

    pub fn ref_current_pos(&self) -> Option<&Pos> {
        self.pos_hist.back()
    }

    pub fn calculate_path(
        &self,
        ref_mut_pixel_map: &mut PixelMap,
        ref_gem_list: &GemList,
    ) -> Option<(Vec<Pos>, u32)> {
        let target_pos = if ref_gem_list.first_known().is_some() {
            let ref_first_gem = ref_gem_list.first_known().unwrap();
            *ref_first_gem.ref_pos()
        } else {
            let pos = find_unknown_pos(ref_mut_pixel_map);
            //eprintln!("unkown_pos: {pos:?}");
            pos
        };
        let current_pos = self.ref_current_pos().unwrap();
        let a_str_path = astar(
            current_pos,
            |p| p.a_star_succesors(ref_mut_pixel_map),
            |p| p.distance(&target_pos),
            |p| *p == target_pos,
        );
        if a_str_path.is_none() && target_pos.x != 0 && target_pos.y != 0 {
            ref_mut_pixel_map[(target_pos.x, target_pos.y)] = PixelType::Wall.as_u8();
            self.calculate_path(ref_mut_pixel_map, ref_gem_list);
        }
        a_str_path
    }

    pub fn get_move(
        &self,
        rng: &mut StdRng,
        ref_mut_pixel_map: &mut PixelMap,
        ref_gem_list: &GemList,
    ) -> &str {
        let current_path = self.current_path(ref_mut_pixel_map, ref_gem_list);
        let opt_next_pos = current_path.get(1);
        let current_pos = self.ref_current_pos().unwrap();
        //let moves = ["N", "S", "E", "W", "WAIT"];
        let moves = ["N", "S", "E", "W"];
        let move_index = rng.random_range(0..moves.len());
        match opt_next_pos {
            Some(next_pos) => {
                let delta_x: i64 = next_pos.x as i64 - current_pos.x as i64;
                let delta_y: i64 = next_pos.y as i64 - current_pos.y as i64;
                //eprintln!("{delta_x},{delta_y}");
                match (delta_x, delta_y) {
                    (-1, 0) => "W",
                    (1, 0) => "E",
                    (0, 1) => "S",
                    (0, -1) => "N",
                    (0, 0) => "WAIT",
                    _ => moves[move_index],
                }
            }
            _ => moves[move_index],
        }
    }

    pub fn current_path(
        &self,
        ref_mut_pixel_map: &mut PixelMap,
        ref_gem_list: &GemList,
    ) -> Vec<Pos> {
        if let Some((path, v)) = self.calculate_path(ref_mut_pixel_map, ref_gem_list) {
            /*for path_pos in &path {
                self.current_path.push_back(*path_pos);
            }*/
            path
        } else {
            Vec::new()
        }
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    //let moves = ["N", "S", "E", "W", "WAIT"];
    let mut first_tick = true;
    let mut gem_list = GemList::new();
    let mut bot = Bot::new();
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
                        PixelType::Unknown.as_u8(),
                    ));
                    first_tick = false;
                }
            }
        }

        if let Some(Value::Object(map)) = data.as_ref() {
            eprintln!("map: {map:?}");
            if let Some(bot_pos_json) = map.get("bot") {
                //eprintln!("Map: {rust_map:?}");
                eprintln!("Bot position: {bot_pos_json:?}");
                let bot_pos = bot_pos_json.try_into().unwrap();
                bot.update_pos(bot_pos);
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
            if let Some(Value::Array(wall_pixels)) = map.get("floor") {
                for floor_pixel in wall_pixels {
                    let floor_pos: Pos = floor_pixel.try_into().unwrap();
                    if let Some(ref mut tmp_map) = opt_map {
                        //tmp_map[(wall_pixel, 0)] = 17;
                        tmp_map[(floor_pos.x, floor_pos.y)] = PixelType::Floor.into();
                    }
                }
            }
            if let Some(Value::Array(channel_meas_vec)) = map.get("channels") {
                eprintln!("channels: {:?}", channel_meas_vec);
                for (channel_id, channel_meas) in channel_meas_vec.iter().enumerate() {
                    let meas = channel_meas.as_f64().unwrap();
                    if meas > 0. {
                        if let Some(ref ref_pixel_map) = opt_map {
                            gem_list.add_gem_with_channel(
                                ref_pixel_map,
                                *bot.ref_current_pos().unwrap(),
                                300,
                                channel_id,
                            );
                        }
                        gem_list.add_channel_measurement(
                            channel_id,
                            *bot.ref_current_pos().unwrap(),
                            meas,
                        );
                    } else {
                        gem_list.remove_channel(channel_id);
                    }
                }
            }

            if let Some(Value::Array(gem_vec_json)) = map.get("visible_gems") {
                for gem_entry_json in gem_vec_json {
                    //eprintln!("GemEntryJson: {gem_entry_json}");
                    let ttl = if let Some(Value::Number(ttl_json)) = gem_entry_json.get("ttl") {
                        ttl_json.as_u64().unwrap()
                    } else {
                        u64::default()
                    };
                    if let Some(gem_pos_json) = gem_entry_json.get("position") {
                        let gem_pos: Pos = gem_pos_json.try_into().unwrap();
                        gem_list.add_gem_with_pos(gem_pos, ttl);
                    }
                }
            }
        }
        gem_list.next_tick(&bot);

        // Emit a random move
        //let move_index = rng.random_range(0..moves.len());
        // Write and flush promptly
        //let highlight_json = "{\"highlight\":[[2,2,\"#00ff0050\"]]}";
        //let pos_vec = vec![Pos { x: 2, y: 2 }];
        let wall_pos_vec = map_to_pos_vec(opt_map.as_ref().unwrap(), PixelType::Wall);
        let mut highlight = Highlight::empty()
            .blue(wall_pos_vec, 70)
            .color(gem_list.known_pos_vec(), "ffff00", 90)
            .color(gem_list.guess_pos_vec(), "ff00ff", 60);
        if let Some(ref_mut_pixel_map) = opt_map.as_mut() {
            let a_star_path = bot.current_path(ref_mut_pixel_map, &gem_list);
            highlight = highlight.white(a_star_path, 90);
            highlight = highlight.color(vec![find_unknown_pos(ref_mut_pixel_map)], "#ff0000", 60);
            let highlight_json = serde_json::to_string(&highlight).unwrap();
            //eprintln!("current_bot_pos: {:?}", bot.ref_current_pos());
            //eprintln!("{gem_list:?}");
            println!(
                "{} {highlight_json}",
                bot.get_move(&mut rng, ref_mut_pixel_map, &gem_list)
            );
            let _ = io::stdout().flush();
        }
    }
}
