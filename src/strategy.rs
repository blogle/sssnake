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
    (ConvBlock<6, 64>, ConvBlock<64, 128>, MaxPool2D<3, 2, 1>),
    (BasicBlock<128>, ReLU),
    (ConvBlock<128, 256>, MaxPool2D<3, 2, 1>),
    (ConvBlock<256, 512>, MaxPool2D<3, 2, 1>),
    (BasicBlock<512>, ReLU),
    (AvgPoolGlobal, Linear<512, NUM_ACTIONS>),
);

#[derive(Clone)]
struct State {
    tensor: [[[f32; 11]; 11]; 6],
}

impl State {
    fn from_gamestate(game_state: crate::GameState) -> Self {
        let mut tensor = [[[f32::NAN; 11]; 11]; 6];
        let snakes = game_state.board.snakes;

        let my_snake = game_state.you;
        let my_health = my_snake.health as f32;

        // Otherwise we need to generate the features that we will feed into the network.
        let mut segment = my_snake.head;
        let (x, y) = (segment.x as usize, 10 - segment.y as usize);
        for pos in &my_snake.body[1..] {
            let (x, y) = (pos.x as usize, 10 - pos.y as usize);
            tensor[0][y][x] = segment.x as f32;
            tensor[1][y][x] = 10.0 - segment.y as f32;
            segment = *pos;
        }

        for pos in &game_state.board.food {
            let (x, y) = (pos.x as usize, 10 - pos.y as usize);
            tensor[4][y][x] = my_health;
        }

        // There is still an opponent on the board
        if let Some(their_snake) = snakes.iter().find(|snake| snake.id != my_snake.id) {
            let their_health = their_snake.health as f32;
            let mut segment = their_snake.head;
            let (x, y) = (segment.x as usize, 10 - segment.y as usize);
            for pos in &their_snake.body[1..] {
                let (x, y) = (pos.x as usize, 10 - pos.y as usize);
                tensor[2][y][x] = segment.x as f32;
                tensor[3][y][x] = 10.0 - segment.y as f32;
                segment = *pos;
            }

            for pos in game_state.board.food {
                let (x, y) = (pos.x as usize, 10 - pos.y as usize);
                tensor[5][y][x] = their_health;
            }
        }

        State { tensor }
    }
}

#[derive(Clone)]
struct Move {
    state: State,
    action: crate::Direction,
}

struct Experience {
    action: Move,
    q_value: f32,
}

type DQN = <ResNet9 as BuildOnDevice<AutoDevice, f32>>::Built;

fn flatten(tensor: &[[[f32; 11]; 11]; 6]) -> &[f32; 6 * 11 * 11] {
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
                experiences.push(sample);
            } else {
                self.buffer.force_push(sample);
            }
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
        const DECAY_RATE: f64 = 0.9995;
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

        let start = std::time::Instant::now();
        for epoch in 1.. {
            let mut total_loss = 0.0;

            // epoch
            for _ in 0..20 {
                let mut state_batch: Vec<f32> = Vec::with_capacity(128 * 6 * 11 * 11);
                let mut action_batch = Vec::with_capacity(128);
                let mut target_batch = Vec::with_capacity(128);

                for Experience {
                    action: Move { state, action },
                    q_value,
                } in self.experiences.sample(128)
                {
                    state_batch.extend(flatten(&state.tensor));
                    action_batch.push(action as usize);
                    target_batch.push(q_value)
                }

                let state_batch_dev: Tensor<Rank4<128, 6, 11, 11>, f32, _> =
                    self.device.tensor(state_batch);
                let action_batch_dev: Tensor<Rank1<128>, usize, _> =
                    self.device.tensor(action_batch);
                let target_batch_dev: Tensor<Rank1<128>, f32, _> = self.device.tensor(target_batch);

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
                loss = total_loss / 20.0,
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

        let x: Tensor<Rank4<30, 6, 11, 11>, f32, _> = dev.ones();
        let start = std::time::Instant::now();
        let samples = 30;
        for _ in 0..samples {
            let _y = m.forward(x.clone());
        }

        println!("Time: {:?}", start.elapsed());
        println!("Avg: {:?}", start.elapsed() / samples);
    }
}
