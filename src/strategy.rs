use dfdx::prelude::*;
use dfdx::optim::{Momentum, Sgd, SgdConfig};
use dfdx::losses::huber_loss;
use rand::Rng;


type ConvBlock<const I: usize, const O: usize> = (
    Conv2D<I, O, 3, 1, 1>,
    BatchNorm2D<O>,
    ReLU,
);

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
    (AvgPoolGlobal, Linear<512, NUM_ACTIONS>)
);

#[derive(Clone)]
enum State {
    Terminal { reward: f32 },
    NonTerminal {
        tensor: [[[f32; 11]; 11]; 6]
    }
}

impl State {
    fn from_gamestate(game_state: crate::GameState) -> Self {
        let mut tensor = [[[f32::NAN; 11]; 11]; 6];
        let snakes = game_state.board.snakes;
        
        let my_snake = game_state.you.unwrap();
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
        if let Some(their_snake) = snakes 
            .iter()
            .find(|snake| snake.id != my_snake.id)

        {
            let their_health = their_snake.health as f32;
            let mut segment = their_snake.head;
            let (x, y) = (segment.x as usize, 10 - segment.y as usize);
            println!("their head {:?}", (x, y));
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

        State::NonTerminal { tensor }
    }
}

#[derive(Clone)]
struct Move {
    state: State,
    action: crate::Direction,
}

#[derive(Default)]
pub struct Game {
    model: Model,
    prior_moves: std::collections::HashMap<String, Move>
}

// Hack to send gpu context across threads
// we should cleanly handle this in the future.
unsafe impl Send for Game {}

type DQN = <ResNet9 as BuildOnDevice<AutoDevice, f32>>::Built;
struct Model {
    device: AutoDevice,
    learner: DQN,
    target: DQN,
    gradients: Gradients<f32, AutoDevice>,
    optimizer: Sgd<DQN, f32, AutoDevice>,
    epsilon: f64,
}

impl Default for Model {
    fn default() -> Self {
        let device = AutoDevice::default();
        let learner = device.build_module::<ResNet9, f32>();
        let gradients = learner.alloc_grads();
		let target = learner.clone();
		let optimizer = Sgd::new(
			&learner,
			SgdConfig {
				lr: 1e-1,
				momentum: Some(Momentum::Nesterov(0.9)),
				weight_decay: None,
			},
		);

        let epsilon = 1.0;
        Model { device, learner, target, gradients, optimizer, epsilon }
    }
}

impl Game {
    pub fn play(&mut self, state: crate::GameState) -> crate::Direction {
        let this_player = state.you.as_ref().unwrap().id.clone();
        let state = State::from_gamestate(state);
        
        let prior_move = self.prior_moves.get(&this_player);
        let new_action = self.generate_move(prior_move.cloned(), state.clone())
            .expect("We have not reached a terminal state so it should be possible to generate a move");

        let _ = self.prior_moves.insert(this_player, Move { state, action: new_action });

        println!("playing {:?}", new_action);
        new_action
    }

    pub fn terminate_game(&mut self, state: crate::GameState) {
        let snakes = state.board.snakes;
        let my_snake = state.you.unwrap();
        // Look through all the snakes on the board and see if we can find our snake,
        // If we arent on the board then we have died and we need to assign a negative reward,
        // if we are the only snake on the board then we have won and we need to assign a positive reward.
        let state = if snakes.iter().find(|snake| snake.id == my_snake.id).is_none() {
            State::Terminal { reward: -100.0 }
        } else {
            State::Terminal { reward: 10.0 }
        };

        // Decay epsilon
        let minimum_epsilon = 0.1;
        let decay_rate = 0.9995;
        if self.model.epsilon > minimum_epsilon {
            self.model.epsilon *= decay_rate;
        }

        let prior_move = self.prior_moves.get(&my_snake.id);
        let None = self.generate_move(prior_move.cloned(), state.clone()) else {
            panic!("We reached a terminal state, we should not have generated a move")
        };
        
        let _ = self.prior_moves.remove(&my_snake.id);

        // Update the target network
        self.model.target.clone_from(&self.model.learner);
    }


    fn generate_move(&mut self, prior_move: Option<Move>, state: State) -> Option<crate::Direction> {
        let model = &mut self.model;
        let mut state_dev: Tensor<Rank4<1, 6, 11, 11>, f32, _> = model.device.ones();

        let mut next_move = None;
        let q_est = match state {
            State::Terminal { reward } => model.device.tensor([reward]),
            State::NonTerminal { ref tensor } => {
                state_dev.copy_from(unsafe {
                    let slice_cast: &[f32; 6 * 11 * 11] = std::mem::transmute(tensor);
                    slice_cast
                });

                // Get the q values for the current state and then select the best action
                let q_values_now = model.target.forward(state_dev.clone());


                let rand_num: f64 = rand::thread_rng().gen_range(0.0..=1.0);
                let action = if rand_num < model.epsilon {
                    rand::thread_rng().gen_range(0..=4)
                } else {
                    let mut argmax = 0;
                    let values = q_values_now.as_vec();
                    for i in 0..4 {
                        if values[i] > values[argmax] {
                            argmax = i;
                        }
                    }

                    argmax
                };

                next_move = Some(match action {
                    0 => crate::Direction::Up,
                    1 => crate::Direction::Down,
                    2 => crate::Direction::Left,
                    _ => crate::Direction::Right,
                });

                // Return our reward estimate
                q_values_now.max::<Rank1<1>, _>() * 0.99
            }
        };

        if let Some(Move { state: State::NonTerminal { ref tensor }, action }) = prior_move {
            // Learn from our last move
            state_dev.copy_from(unsafe {
                let slice_cast: &[f32; 6 * 11 * 11] = std::mem::transmute(tensor);
                slice_cast
            });

            let state_dev_t: Tensor<Rank4<1, 6, 11, 11>, f32, _, OwnedTape<f32, _>> = state_dev.traced(model.gradients.to_owned());
            let q_values_prev = model.learner.forward_mut(state_dev_t);

            let action: Tensor<Rank1<1>, usize, _> = model.device.tensor([action as usize]);
            let action_q = q_values_prev.select(action);
            let loss = huber_loss(action_q, q_est, 1.0);

            model.gradients = loss.backward();
            model.optimizer.update(&mut model.learner, &mut model.gradients).expect("Unused params");
            model.learner.zero_grads(&mut model.gradients);
        }

        next_move
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
