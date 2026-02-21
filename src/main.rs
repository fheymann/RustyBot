use ndarray::prelude::*;
use ndarray::{Array, Ix2};
use pathfinding::prelude::astar;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::Value;
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

#[derive(Debug, Clone)]
pub struct ScrimConfig {
    pub bot_seed: u64,
    pub width: u64,
    pub height: u64,
    pub emit_signal_channels: bool,
    pub enable_debug: bool,
    pub gem_spawn_rate: f64,
    pub gem_ttl: u64,
    pub generator: String,
    pub max_gems: u64,
    pub max_ticks: u64,
    pub signal_cutoff: f64,
    pub signal_fade: f64,
    pub signal_noise: f64,
    pub signal_quantization: f64,
    pub signal_radius: f64,
    pub stage_key: String,
    pub timeout_scale: f64,
    pub vis_radius: f64,
}

impl Default for ScrimConfig {
    fn default() -> Self {
        Self {
            bot_seed: 0,
            width: 0,
            height: 0,
            emit_signal_channels: false,
            enable_debug: true,
            gem_spawn_rate: 0.,
            gem_ttl: 300,
            generator: "cellular".to_string(),
            max_gems: 0,
            max_ticks: 0,
            signal_cutoff: 0.,
            signal_fade: 0.,
            signal_noise: 0.,
            signal_quantization: 0.,
            signal_radius: 0.,
            stage_key: "stage".to_string(),
            timeout_scale: 0.,
            vis_radius: 0.,
        }
    }
}

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
                (xy_array.first(), xy_array.get(1))
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

    pub fn best_target_pos(&self, ref_scrim_config: &ScrimConfig, ref_bot: &Bot) -> Option<Pos> {
        let mut ret_pos = None;
        if !self.known_pos_vec().is_empty() {
            let mut min_dist = u32::MAX;
            for pos in self.known_pos_vec() {
                let dist = pos.distance(&ref_bot.ref_current_pos().unwrap());
                if dist < min_dist {
                    min_dist = dist;
                    ret_pos = Some(pos);
                }
            }
        } else if !self.guess_pos_vec().is_empty() {
            let mut max_signal = f64::MIN;
            for gem in self.guess_vec() {
                if gem.ttl < ref_scrim_config.gem_ttl - ref_scrim_config.signal_fade as u64
                    && let Some(tmp_signal) = ref_bot.get_signal(gem.channel_id())
                {
                    if *tmp_signal > max_signal {
                        max_signal = *tmp_signal;
                        ret_pos = Some(gem.guess_pos);
                    }
                }
            }
        }
        ret_pos
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

    pub fn remove_gem_with_channel(&mut self, channel_id: usize) {
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

    pub fn ref_mut_gem_by_pos(&mut self, pos: Pos) -> Option<&mut Gem> {
        let mut ref_gem = None;
        for gem in &mut self.gem_vec {
            if *gem.ref_pos() == pos {
                ref_gem = Some(gem);
                break;
            }
        }
        ref_gem
    }

    pub fn add_channel_measurement(
        &mut self,
        ref_scrim_config: &ScrimConfig,
        channel_id: usize,
        bot_pos: Pos,
        signal: f64,
    ) {
        if let Some(ref mut ref_gem) = self.ref_mut_gem_by_channel(channel_id) {
            //eprintln!("add_measurement");
            ref_gem.add_measurement(ref_scrim_config, bot_pos, signal);
        }
    }

    pub fn add_gem_with_pos(&mut self, pos: Pos, ttl: u64) {
        if !self.exists_pos(&pos) {
            self.gem_vec.push(Gem::new_with_pos(pos, ttl));
        } else {
            let gem = self.ref_mut_gem_by_pos(pos).unwrap();
            gem.known_pos = Some(pos);
            gem.ttl = ttl;
            //eprintln!("Gem exists at pos: {pos:?} with ttl: {ttl}");
        }
    }

    pub fn add_gem_with_channel(&mut self, ref_pixel_map: &PixelMap, ttl: u64, channel_id: usize) {
        if !self.exists_channel(channel_id) {
            self.gem_vec
                .push(Gem::new_with_channel(ref_pixel_map, ttl, channel_id));
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

    pub fn next_tick(&mut self, ref_scrim_config: &ScrimConfig, ref_bot: &Bot, ref_pixel_map: &PixelMap) {
        self.check_bot_pos(ref_bot);
        for gem in &mut self.gem_vec {
            gem.next_tick(ref_scrim_config, ref_bot,ref_pixel_map);
        }
    }

    pub fn first_guess(&self) -> Option<Gem> {
        if !self.guess_vec().is_empty() {
            Some(self.guess_vec()[0].clone())
        } else {
            None
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

    pub fn is_empty(&self) -> bool {
        self.gem_vec.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct Gem {
    known_pos: Option<Pos>,
    guess_pos: Pos,
    guess_err: f64,
    guess_not_moved: usize,
    ttl: u64,
    meas_hist: VecDeque<(Pos, f64, f64)>,
    channel_id: usize,

}

impl Gem {
    pub fn new_with_channel(ref_pixel_map: &PixelMap, ttl: u64, channel_id: usize) -> Self {
        let guess_pos = Pos::new(ref_pixel_map.dim().0 / 2, ref_pixel_map.dim().1 / 2);
        Self {
            known_pos: None,
            ttl,
            guess_pos,
            guess_err: f64::MAX,
            channel_id,
            meas_hist: VecDeque::new(),
            guess_not_moved: 0,
        }
    }

    pub fn new_with_pos(pos: Pos, ttl: u64) -> Self {
        Self {
            known_pos: Some(pos),
            ttl,
            guess_pos: pos,
            guess_err: 0.,
            channel_id: usize::MAX,
            meas_hist: VecDeque::new(),
            guess_not_moved:0 
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

    pub fn add_measurement(&mut self, ref_scrim_config: &ScrimConfig, bot_pos: Pos, signal: f64) {
        let signal_fade = ref_scrim_config.signal_fade;
        let tmp_fade = (ref_scrim_config.gem_ttl + 1 - self.ttl) as f64 / (signal_fade);
        let fade = if tmp_fade < 1. { tmp_fade } else { 1. };
        //eprintln!("adding measurement fade: {fade} signal:{signal}");
        self.meas_hist.push_back((bot_pos, fade, signal));
        /*if self.guess_err > 1e-5 {
            self.meas_hist.push_back((bot_pos, fade, signal));
        }
        if self.meas_hist.len() > 50 {
            self.meas_hist.pop_front();
        }*/
    }

    pub fn next_tick(&mut self, ref_scrim_config: &ScrimConfig, ref_bot: &Bot,ref_pixel_map: &PixelMap) {
        self.ttl -= 1;
        self.guess_err*=f64::MAX;
        if self.known_pos.is_none() {
            let current_guess = self.guess_pos;
            let mut new_guess_x = current_guess.x;
            let mut new_guess_y = current_guess.y;
            let ix_vec: Vec<i64> = (-32..32).collect();
            let iy_vec: Vec<i64> = (-32..32).collect();
            let signal_radius = ref_scrim_config.signal_radius;
            let mut moved = true;
            for ix in &ix_vec {
                for iy in &iy_vec {
                    let new_x = current_guess.x as i64 + *ix;
                    let new_y = current_guess.y as i64 + *iy;
                    let mut tmp_min = 0.;
                    for (meas_pos, fade, meas_signal) in &self.meas_hist {
                        let delta_x = (new_x - meas_pos.x as i64) as f64;
                        let delta_y = (new_y - meas_pos.y as i64) as f64;
                        let distance = (delta_x.powf(2.) + delta_y.powf(2.)).sqrt();
                        let gem_signal: f64 = fade / (1. + (distance / signal_radius).powf(2.0));
                        tmp_min += (gem_signal - meas_signal).powf(2.);
                    }
                    //tmp_min = tmp_min / self.meas_hist.len() as f64;
                    if tmp_min <= self.guess_err*0.99 {
                        if *ix==0 && *iy==0 {
                            moved = false
                        } else {
                            moved = true
                        }
                        self.guess_err = tmp_min;
                        new_guess_x = new_x as usize;
                        new_guess_y = new_y as usize;
                        //eprintln!("found new min: {} @ {new_guess_x},{new_guess_y}",self.guess_err);
                    }
                }
            }
            if new_guess_x > ref_scrim_config.width as usize - 1 {
                new_guess_x = ref_scrim_config.width as usize - 1;
            }
            if new_guess_y > ref_scrim_config.height as usize - 1 {
                new_guess_y = ref_scrim_config.height as usize - 1;
            }
            self.guess_pos = Pos::new(new_guess_x, new_guess_y);
            if !moved {
            self.guess_not_moved +=1;
            } else {
                self.guess_not_moved = 0;
            }
            if self.guess_not_moved as f64>ref_scrim_config.signal_fade {
                //eprintln!("new_x: {new_guess_x} new_y: {new_guess_y} tmp_min:{}",self.guess_err);
                if !is_pixel_type(&self.guess_pos, ref_pixel_map,PixelType::Wall) {
                self.known_pos = Some(self.guess_pos)
                }
            } //found_tmp_min = true;
            /*eprintln!(
                "guessing new Pos: {:?}, error: {} noise {}",
                self.guess_pos,
                self.guess_err,
                ref_scrim_config.signal_noise * 2.
            );*/
        }
    }
}

pub type PixelMap = Array<u8, Ix2>;

pub fn is_pixel_type(ref_pos:&Pos,ref_pixel_map: &PixelMap,pixel_type:PixelType) -> bool {
    let pix = *ref_pixel_map.get((ref_pos.x,ref_pos.y)).unwrap();
    pix == pixel_type.as_u8()
}

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
    target_pos: Option<Pos>,
    current_path: VecDeque<Pos>,
    current_path_value: u32,
    signal_vec: Vec<f64>,
}

impl Bot {
    pub fn new() -> Self {
        Self {
            pos_hist: VecDeque::new(),
            target_pos: None,
            current_path: VecDeque::new(),
            current_path_value: 0,
            signal_vec: Vec::new(),
        }
    }

    pub fn update_pos(&mut self, pos: Pos) {
        self.pos_hist.push_back(pos);
        /*if self.pos_hist.len() > 100 {
            self.pos_hist.pop_front();
        }*/
    }

    pub fn get_signal(&self, channel_id: usize) -> Option<&f64> {
        self.signal_vec.get(channel_id)
    }

    pub fn update_signal(&mut self, signal: f64, channel_id: usize) {
        while self.signal_vec.len() < channel_id + 1 {
            self.signal_vec.push(f64::default());
        }
        self.signal_vec[channel_id] = signal;
    }

    pub fn ref_current_pos(&self) -> Option<&Pos> {
        self.pos_hist.back()
    }

    pub fn calculate_path(
        &mut self,
        ref_scrim_config: &ScrimConfig,
        ref_mut_pixel_map: &mut PixelMap,
        ref_gem_list: &GemList,
    ) -> Option<(Vec<Pos>, u32)> {
        let target_pos = match ref_gem_list.best_target_pos(ref_scrim_config, &self) {
            Some(pos) => pos,
            None => {
                let mut pos = find_unknown_pos(ref_mut_pixel_map);
                if pos==Pos::new(0,0) {
                    let mid_x = (ref_scrim_config.width/2) as i64;
                    let mid_y = (ref_scrim_config.height/2) as i64;
                    let ix_vec: Vec<i64> = (-32..32).collect();
                    let iy_vec: Vec<i64> = (-32..32).collect();
                    let min_dist = 65;
                    for ix in &ix_vec {
                        for iy in &iy_vec {
                            let dist = ix.abs() + iy.abs();
                            let tmp_pos = Pos::new((mid_x+ix) as usize,(mid_y+iy) as usize); 
                            if dist<min_dist && !is_pixel_type(&tmp_pos, ref_mut_pixel_map, PixelType::Wall){
                                pos = tmp_pos
                            }
                        }
                    }
                }
                pos
            },
        };
        if self.target_pos.is_none() {
            //eprintln!("target_pos is none");
            let current_pos = self.ref_current_pos().unwrap();
            let mut a_str_path = astar(
                current_pos,
                |p| p.a_star_succesors(ref_mut_pixel_map),
                |p| p.distance(&target_pos),
                |p| *p == target_pos,
            );
            for _i in 0..32 {
                if a_str_path.is_none() {
                    //eprintln!("looping {}", i);
                    if target_pos.x > 0
                        && target_pos.x < ref_mut_pixel_map.dim().0 - 1
                        && target_pos.y > 0
                        && target_pos.y < ref_mut_pixel_map.dim().1 - 1
                    {
                        ref_mut_pixel_map[(target_pos.x, target_pos.y)] = PixelType::Wall.as_u8();
                        a_str_path = astar(
                            current_pos,
                            |p| p.a_star_succesors(ref_mut_pixel_map),
                            |p| p.distance(&target_pos),
                            |p| *p == target_pos,
                        );
                    }
                }
            }
            if a_str_path.is_some() {
                let tmp = a_str_path.as_ref().unwrap();
                self.current_path_value = tmp.1;
                self.current_path = VecDeque::new();
                for pos in &tmp.0 {
                    self.current_path.push_back(*pos);
                }
                self.current_path.pop_front();
                self.target_pos = Some(target_pos);
                a_str_path
            } else {
                if target_pos.x > 0
                    && target_pos.x < ref_mut_pixel_map.dim().0 - 1
                    && target_pos.y > 0
                    && target_pos.y < ref_mut_pixel_map.dim().1 - 1
                {
                        ref_mut_pixel_map[(target_pos.x, target_pos.y)] = PixelType::Wall.as_u8();
                }
                self.current_path = VecDeque::new();
                self.current_path_value = 0;
                self.target_pos = None;
                None
            }
        } else {
            let mut a_star_vec = Vec::new();
            for pos in &self.current_path {
                a_star_vec.push(*pos);
            }
            Some((a_star_vec, self.current_path_value))
        }
    }

    pub fn reset_path(
        &mut self,
        ref_scrim_config: &ScrimConfig,
        ref_mut_pixel_map: &mut PixelMap,
        ref_gem_list: &GemList,
    ) {
        self.target_pos = None;
        self.current_path = VecDeque::new();
        self.calculate_path(ref_scrim_config, ref_mut_pixel_map, ref_gem_list);
    }

    pub fn get_move(
        &mut self,
        rng: &mut StdRng,
        ref_scrim_config: &ScrimConfig,
        ref_mut_pixel_map: &mut PixelMap,
        ref_gem_list: &GemList,
    ) -> &str {
        let current_pos = *self.ref_current_pos().unwrap();
        let opt_next_pos = self.current_path.pop_front();
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
                    //(0, 0) => "WAIT",
                    _ => {
                        self.target_pos = Some(find_unknown_pos(ref_mut_pixel_map));
                        self.current_path = VecDeque::new();
                        self.calculate_path(ref_scrim_config, ref_mut_pixel_map, ref_gem_list);
                        if !self.current_path.is_empty() {
                            self.get_move(rng, ref_scrim_config, ref_mut_pixel_map, ref_gem_list)
                        } else {
                            moves[move_index]
                        }
                    }
                }
            }
            _ => {
                self.reset_path(ref_scrim_config, ref_mut_pixel_map, ref_gem_list);
                if !self.current_path.is_empty() {
                    self.get_move(rng, ref_scrim_config, ref_mut_pixel_map, ref_gem_list)
                } else {
                    moves[move_index]
                }
            }
        }
    }

    pub fn current_path(
        &mut self,
        ref_scrim_config: &ScrimConfig,
        ref_mut_pixel_map: &mut PixelMap,
        ref_gem_list: &GemList,
    ) -> Vec<Pos> {
        if let Some((path, _v)) =
            self.calculate_path(ref_scrim_config, ref_mut_pixel_map, ref_gem_list)
        {
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
    let mut scrim_config = ScrimConfig::default();
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
                    //eprintln!("{cfg_map:?}");
                    let bot_seed = cfg_map
                        .get("bot_seed")
                        .map(|v| v.as_u64().unwrap_or(0))
                        .unwrap_or(0);
                    scrim_config.bot_seed = bot_seed;
                    let gem_ttl = cfg_map
                        .get("gem_ttl")
                        .map(|v| v.as_u64().unwrap_or(0))
                        .unwrap_or(0);
                    scrim_config.gem_ttl = gem_ttl;
                    let signal_radius = cfg_map
                        .get("signal_radius")
                        .map(|v| v.as_f64().unwrap_or(0.))
                        .unwrap_or(0.);
                    scrim_config.signal_radius = signal_radius;
                    let signal_noise = cfg_map
                        .get("signal_noise")
                        .map(|v| v.as_f64().unwrap_or(0.))
                        .unwrap_or(0.);
                    scrim_config.signal_noise = signal_noise;
                    let signal_fade = cfg_map
                        .get("signal_fade")
                        .map(|v| v.as_f64().unwrap_or(0.))
                        .unwrap_or(0.);
                    scrim_config.signal_fade = signal_fade;

                    let width = cfg_map
                        .get("width")
                        .map(|v| v.as_u64().unwrap_or(0))
                        .unwrap_or(0);
                    scrim_config.width = width;
                    let height = cfg_map
                        .get("height")
                        .map(|v| v.as_u64().unwrap_or(0))
                        .unwrap_or(0);
                    scrim_config.height = height;
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
            //eprintln!("map: {map:?}");
            if let Some(bot_pos_json) = map.get("bot") {
                //eprintln!("Map: {rust_map:?}");
                //eprintln!("Bot position: {bot_pos_json:?}");
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
                //eprintln!("channels: {:?}", channel_meas_vec);
                for (channel_id, channel_meas) in channel_meas_vec.iter().enumerate() {
                    let meas = channel_meas.as_f64().unwrap();
                    if meas > 0. {
                        if let Some(ref ref_pixel_map) = opt_map {
                            gem_list.add_gem_with_channel(ref_pixel_map, 300, channel_id);
                        }
                        gem_list.add_channel_measurement(
                            &scrim_config,
                            channel_id,
                            *bot.ref_current_pos().unwrap(),
                            meas,
                        );
                        bot.update_signal(meas, channel_id);
                    } else {
                        gem_list.remove_gem_with_channel(channel_id);
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
                        //eprintln!("visible_gem: {gem_pos:?}");
                    }
                }
                if let Some(ref mut ref_mut_pixel_map) = opt_map {
                    bot.reset_path(&scrim_config, ref_mut_pixel_map, &gem_list);
                }
            }
        }
        if let Some(ref ref_pixel_map) = opt_map {
            gem_list.next_tick(&scrim_config, &bot,ref_pixel_map);
        }

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
            let a_star_path = bot.current_path(&scrim_config, ref_mut_pixel_map, &gem_list);
            highlight = highlight.white(a_star_path, 90);
            highlight = highlight.color(vec![find_unknown_pos(ref_mut_pixel_map)], "#ff0000", 60);
            let highlight_json = serde_json::to_string(&highlight).unwrap();
            //eprintln!("current_bot_pos: {:?}", bot.ref_current_pos());
            //eprintln!("{gem_list:?}");
            println!(
                "{} {highlight_json}",
                bot.get_move(&mut rng, &scrim_config, ref_mut_pixel_map, &gem_list)
            );
            let _ = io::stdout().flush();
        }
    }
}
