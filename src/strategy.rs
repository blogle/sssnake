use dashmap::DashMap;
use dfdx::data::IteratorStackExt;
use dfdx::losses::huber_loss;
use dfdx::optim::{Momentum, Sgd, SgdConfig};
use dfdx::prelude::*;
use rand::Rng;
use std::sync::{atomic, RwLock};

type ConvBlock<const I: usize, const O: usize> = (Conv2D<I, O, 3, 1, 1>, BatchNorm2D<O>, ReLU);

type BasicBlock<const C: usize> = Residual<(
    Conv2D<C, C, 3, 1, 1>,
    BatchNorm2D<C>,
    ReLU,
    Conv2D<C, C, 3, 1, 1>,
    BatchNorm2D<C>,
)>;

const NUM_ACTIONS: usize = 4;
type ResNet9 = (
    (ConvBlock<F_C, 64>, ConvBlock<64, 128>, MaxPool2D<3, 2, 1>),
    (BasicBlock<128>, ReLU),
    (ConvBlock<128, 256>, MaxPool2D<3, 2, 1>),
    (ConvBlock<256, 512>, MaxPool2D<3, 2, 1>),
    (BasicBlock<512>, ReLU),
    (AvgPoolGlobal, Linear<512, NUM_ACTIONS>),
);

type DQN = <ResNet9 as BuildOnDevice<AutoDevice, f32>>::Built;

const HEIGHT: usize = 4;
const WIDTH: usize = 4;
const F_H: usize = HEIGHT + 2;
const F_W: usize = WIDTH + 2;
const F_C: usize = 6;

#[derive(Clone)]
struct State {
    tensor: [[[f32; F_W]; F_H]; F_C],
}

fn coord_idx(coord: &crate::Coordinate) -> (usize, usize) {
    let (px, py) = (coord.x + 1, coord.y + 1);
    (px as usize, py as usize)
}

impl State {

    fn from_gamestate(game_state: crate::GameState) -> Self {
        // Features
        // Channel 0: one-hot encoding of food positions
        // Channel 1: one-hot encoding of my snakes head position
        // Channel 2: one-hot encoding of oponent snakes head posiitions 
        // Channel 3: one-hot encoding of any obstacles on the board (walls, tails)
        // Channel 4-5: vector field of obstacle movement
        // Note: The height and width are padded so we can model the edges of the board as obstacles,
        // also the y axis is inverted (0, 0) refers to the bottom left cell of the board
        
        let mut tensor = [[[0.0; F_W]; F_H]; F_C];
        let my_snake = game_state.you;
        let snakes = game_state.board.snakes;

        // Populate channel 0 of the tensor - one-hot encoding of food positions
        for (x, y) in game_state.board.food.iter().map(coord_idx) {
            tensor[0][y][x] = 1.0;
        }

        for snake in &snakes {
            let mut next_pos = snake.head;
            let (x, y) = coord_idx(&next_pos);
            if snake.id == my_snake.id {
                // Populate channel 1 of the tensor - one-hot encoding of my snakes head position
                tensor[1][y][x] = 1.0;
            } else {
                // Populate channel 2 of the tensor - one-hot encoding of opponent snakes head positions
                tensor[2][y][x] = 1.0;
            }

            // Populate channels 3, 4 and 5 with the snake tail obstacles and directions
            for pos in &snake.body[1..] {

                // The battlesnake cli gives us duplicate segments
                if *pos == next_pos { 
                    break
                };

                let (x, y) = coord_idx(pos);
                tensor[3][y][x] = 1.0;
                tensor[4][y][x] = (next_pos.x as isize - pos.x as isize) as f32;
                tensor[5][y][x] = (next_pos.y as isize - pos.y as isize) as f32;
                next_pos = *pos;
            }
        }

        // Populate channel 3 of the tensor with the walls
        // Top and bottom walls
        for x in 0..F_W {
            tensor[3][0][x] = 1.0;
            tensor[3][F_H - 1][x] = 1.0;
        }

        // Left and right walls
        for y in 0..F_H {
            tensor[3][y][0] = 1.0;
            tensor[3][y][F_W - 1] = 1.0;
        }

        State { tensor }
    }

}

#[derive(Clone)]
struct Move {
    state: State,
    action: crate::Direction,
}

#[derive(Clone)]
struct Experience {
    action: Move,
    q_value: f32,
}


fn flatten(tensor: &[[[f32; F_W]; F_H]; F_C]) -> &[f32; F_C * F_H * F_W] {
    unsafe { std::mem::transmute(tensor) }
}

struct ReplayBuffer {
    buffer: crossbeam::queue::ArrayQueue<Experience>,
}

impl ReplayBuffer {
    fn new() -> Self {
        ReplayBuffer {
            buffer: crossbeam::queue::ArrayQueue::new(1000),
        }
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn push(&self, experience: Experience) {
        let _ = self.buffer.force_push(experience);
    }

    fn pop(&self) -> Experience {
        let backoff = crossbeam::utils::Backoff::new();
        loop {
            match self.buffer.pop() {
                None if backoff.is_completed() => std::thread::park(),
                None => backoff.spin(),
                Some(experience) => return experience,
            }
        }
    }

    fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut experiences = Vec::with_capacity(batch_size);
        while experiences.len() < batch_size {
            let sample = self.pop();
            if rand::random() {
                experiences.push(sample.clone());
            }

            self.push(sample);
        }

        experiences
    }
}

pub struct Agent {
    device: AutoDevice,
    model: RwLock<DQN>,
    prior_moves: DashMap<String, Move>,
    epsilon: atomic::AtomicU64,
    experiences: ReplayBuffer,
}

// It is unclear that accessing the device across threads is safe.
// All of its fields are ARC'd, but it is not marked as Send or Sync.
unsafe impl Send for Agent {}
unsafe impl Sync for Agent {}

impl Default for Agent {
    fn default() -> Self {
        let device = AutoDevice::default();
        let model = device.build_module::<ResNet9, f32>();
        Self {
            device: device,
            model: RwLock::new(model),
            prior_moves: DashMap::new(),
            epsilon: atomic::AtomicU64::new(1.0f64.to_bits()),
            experiences: ReplayBuffer::new(),
        }
    }
}

impl Agent {
    fn epsilon(&self) -> f64 {
        f64::from_bits(self.epsilon.load(atomic::Ordering::Relaxed))
    }

    fn decay_epsilon(&self) {
        const DECAY_RATE: f64 = 0.9;
        const MINIMUM_EPSILON: f64 = 0.05;

        let epsilon = self.epsilon();
        if epsilon > MINIMUM_EPSILON {
            self.epsilon.store(
                MINIMUM_EPSILON.max(epsilon * DECAY_RATE).to_bits(),
                atomic::Ordering::Relaxed,
            );
        }
    }

    pub fn play(&self, state: crate::GameState) -> crate::Direction {
        let my_snake = state.you.id.clone();
        let state = State::from_gamestate(state);

        let model = self.model.read().unwrap();

        // Get the q values for the current state
        let state_dev = self.device.tensor(state.tensor);
        let q_values = model.forward(state_dev).as_vec();

        // Get the argmax of the q values
        let mut arg_max = 0;
        for i in 0..4 {
            if q_values[i] > q_values[arg_max] {
                arg_max = i;
            }
        }

        // Epsilon Greedy
        let rand_num = rand::thread_rng().gen_range(0.0..=1.0);
        let action_id = if rand_num < self.epsilon() {
            // Explore by taking a random action
            rand::thread_rng().gen_range(0..4)
        } else {
            // Exploit by playing the argmax of our q values
            arg_max
        };

        let action = match action_id {
            0 => crate::Direction::Up,
            1 => crate::Direction::Down,
            2 => crate::Direction::Left,
            _ => crate::Direction::Right,
        };

        // Update keep a reference of the move we are taking with this snake, so that on the next
        // turn when we have the q valuest for the resulting state we can add it to our replay
        // buffer
        if let Some(prior_move) = self.prior_moves.insert(my_snake, Move { state, action }) {
            const GAMMA: f32 = 0.99;
            // Now that we have the target max Q in St we can add Q(St-1, At-1) to the replay buffer
            self.experiences.push(Experience {
                action: prior_move,
                q_value: 1.0 + GAMMA * q_values[arg_max],
            });
        }

        action
    }

    pub fn terminate_episode(&self, state: crate::GameState) {
        let my_snake = state.you;
        if let Some((_, prior_move)) = self.prior_moves.remove(&my_snake.id) {
            let active_snakes = state.board.snakes;

            // If we are the only snake on the board then we have won and we need to assign a positive reward.
            // If we arent on the board then we have died and we need to assign a negative reward,
            let reward = match active_snakes
                .into_iter()
                .find(|snake| snake.id == my_snake.id)
            {
                Some(_) => 1.0, // Winning is nice
                None => -100.0,  // Losing is the awful
            };

            // Add terminal state to the replay buffer
            self.experiences.push(Experience {
                action: prior_move,
                q_value: reward,
            });
        }
    }

    fn clone_model(&self) -> DQN {
        let model = self.model.read().unwrap();
        model.clone()
    }

    fn update_model(&self, new_model: &DQN) {
        self.model.write().unwrap().clone_from(new_model);
    }

    pub fn train(self: std::sync::Arc<Self>) {
        let mut learner = self.clone_model();

        let mut optimizer = Sgd::new(
            &learner,
            SgdConfig {
                lr: 1e-1,
                momentum: Some(Momentum::Nesterov(0.9)),
                weight_decay: None,
            },
        );

        let mut grads = learner.alloc_grads();
        const BATCH_SIZE: usize = 128;
        const EPISODES: f32 = 20.0;

        let start = std::time::Instant::now();
        for epoch in 1.. {
            let mut total_loss = 0.0;

            // epoch
            for _ in 0..EPISODES as usize {
                let mut state_batch: Vec<f32> = Vec::with_capacity(BATCH_SIZE * F_C * F_H * F_W);
                let mut action_batch = Vec::with_capacity(BATCH_SIZE);
                let mut target_batch = Vec::with_capacity(BATCH_SIZE);

                for Experience {
                    action: Move { state, action },
                    q_value,
                } in self.experiences.sample(BATCH_SIZE)
                {
                    state_batch.extend(flatten(&state.tensor));
                    action_batch.push(action as usize);
                    target_batch.push(q_value)
                }

                let state_batch_dev: Tensor<Rank4<BATCH_SIZE, F_C, F_H, F_H>, f32, _> =
                    self.device.tensor(state_batch);
                let action_batch_dev: Tensor<Rank1<BATCH_SIZE>, usize, _> =
                    self.device.tensor(action_batch);
                let target_batch_dev: Tensor<Rank1<BATCH_SIZE>, f32, _> = self.device.tensor(target_batch);

                let q_values_dev = learner.forward_mut(state_batch_dev.traced(grads.to_owned()));
                let action_q = q_values_dev.select(action_batch_dev);

                let loss = huber_loss(action_q, target_batch_dev, 1.0);
                total_loss += loss.array();

                grads = loss.backward();
                optimizer
                    .update(&mut learner, &grads)
                    .expect("Unused params");
                learner.zero_grads(&mut grads);
            }

            tracing::info!(
                epoch = epoch,
                loss = total_loss / EPISODES,
                elapsed = ?start.elapsed(),
            );

            // Update the target network
            self.update_model(&learner);

            // Decay epsilon
            self.decay_epsilon();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn forward_pass() {
        let dev = AutoDevice::default();
        let m = dev.build_module::<ResNet9, f32>();

        let x: Tensor<Rank4<30, F_C, F_H, F_W>, f32, _> = dev.ones();
        let start = std::time::Instant::now();
        let samples = 30;
        for _ in 0..samples {
            let _y = m.forward(x.clone());
        }

        println!("Time: {:?}", start.elapsed());
        println!("Avg: {:?}", start.elapsed() / samples);
    }
}
