use anyhow::{
    Context, 
    Error
};
use axum::{
    extract::State,
    routing::{get, post},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json, Router,
};
use tower_http::{
    classify::ServerErrorsFailureClass,
    trace::TraceLayer
};

use tracing_subscriber::{filter, prelude::*};

use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct BotError(Error);


#[derive(serde::Serialize)]
struct BotInfo {
    apiversion: String,
    author: String,
    color: String,
    head: String,
    tail: String,
    version: String
}


async fn bot_info() -> (StatusCode, Json<BotInfo>) {
    let info = BotInfo {
        apiversion: String::from("1"),
        author: String::from("ogle"),
        color: String::from("#FF0000"),
        head: String::from("default"),
        tail: String::from("default"),
        version: String::from("0.0.1")
    };

    (StatusCode::OK, Json(info))
}

async fn initialize_state(Json(_payload): Json<sssnake::GameState>) {
}

async fn generate_move(
    State(game): State<Arc<Mutex<sssnake::strategy::Game>>>,
    Json(state): Json<sssnake::GameState>
) -> Json<sssnake::Move>
{
    let mut guard = game.lock().unwrap();

    Json(sssnake::Move { 
        shout: String::from("I'm a slippperry lil ssssnake!"),
        direction: guard.play(state), 
    })
}

async fn terminate_game(
    State(game): State<Arc<Mutex<sssnake::strategy::Game>>>,
    Json(state): Json<sssnake::GameState>
)
{
    let mut guard = game.lock().unwrap();
    guard.terminate_game(state);
}

async fn agent_server() -> Result<(), Error> {
    let agent_state = std::sync::Arc::new(
        std::sync::Mutex::new(
            sssnake::strategy::Game::default()
        )
    );

    let app = Router::new()
        .route("/", get(bot_info))
        .route("/start", post(initialize_state))
        .route("/move", post(generate_move))
        .route("/end", post(terminate_game))
        .layer(TraceLayer::new_for_http())
        .with_state(agent_state);

    let sock_addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    axum::Server::bind(&sock_addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}


#[tokio::main]
async fn main() -> Result<(), Error> {
    // initialize tracing
    let targets = filter::Targets::new()
        .with_target("sssnake", tracing::Level::DEBUG)
        .with_target("tower_http", tracing::Level::DEBUG)
        .with_target("axum::rejection", tracing::Level::TRACE);

    tracing_subscriber::registry()
        .with(targets)
        .with(tracing_subscriber::fmt::layer())
        .init();

    agent_server().await?;

    Ok(())
}
