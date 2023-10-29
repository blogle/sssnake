#![feature(generic_const_exprs)]
use serde::{Deserialize, Serialize};

pub mod board_api;
pub mod db;
pub mod strategy;

#[derive(Debug, Deserialize)]
pub struct Ruleset {
    name: String,
    version: String,
}

#[derive(Debug, Deserialize)]
pub struct Game {
    id: String,
    ruleset: Ruleset,
    timeout: u32,
    source: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct Coordinate {
    x: isize,
    y: isize,
}

#[derive(Deserialize)]
pub struct Board {
    height: usize,
    width: usize,
    food: Vec<Coordinate>,
    hazards: Vec<Coordinate>,
    pub snakes: Vec<Snake>,
}

#[derive(Debug, Deserialize)]
pub struct Snake {
    id: String,
    name: String,
    health: u32,
    body: Vec<Coordinate>,
    latency: String,
    head: Coordinate,
    length: usize,
    shout: String,
}

#[derive(Deserialize)]
pub struct GameState {
    game: Game,
    turn: usize,
    pub board: Board,
    you: Snake,
}

#[derive(Serialize, Clone, Copy, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

#[derive(Debug, Serialize)]
pub struct Move {
    #[serde(rename = "move")]
    pub direction: Direction,
    pub shout: String,
}
