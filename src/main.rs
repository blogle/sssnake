use anyhow::{Context, Error};
use axum::{
    extract::{ws, ConnectInfo, Path, Query, State},
    http,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::prelude::*;
use tower_http::{classify::ServerErrorsFailureClass, cors::CorsLayer, trace::TraceLayer};

use tracing_subscriber::{filter, prelude::*};
use tungstenite::protocol::Message;

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

#[derive(Debug)]
struct BotError(Error);

#[derive(serde::Serialize)]
struct BotInfo {
    apiversion: String,
    author: String,
    color: String,
    head: String,
    tail: String,
    version: String,
}

async fn bot_info() -> (http::StatusCode, Json<BotInfo>) {
    let info = BotInfo {
        apiversion: String::from("1"),
        author: String::from("ogle"),
        color: String::from("#FF0000"),
        head: String::from("default"),
        tail: String::from("default"),
        version: String::from("0.0.1"),
    };

    (http::StatusCode::OK, Json(info))
}

async fn initialize_state(Json(_payload): Json<sssnake::GameState>) {}

async fn generate_move(
    State(app): State<Arc<AppState>>,
    Json(state): Json<sssnake::GameState>,
) -> Json<sssnake::Move> {
    let agent = &app.agent;
    Json(sssnake::Move {
        shout: String::from("I'm a slippperry lil ssssnake!"),
        direction: agent.play(state),
    })
}

async fn terminate_game(State(app): State<Arc<AppState>>, Json(state): Json<sssnake::GameState>) {
    app.agent.terminate_episode(state);
}

async fn game_event_metadata(
    State(app): State<Arc<AppState>>,
    Path(game_id): Path<String>,
) -> impl IntoResponse {
    println!("finding the event for game {}", game_id);
    let event = app.db.get_game_metadata(&game_id).unwrap_or_else(|err| {
        tracing::error!("Failed to query game {}: {}", game_id, err);
        None
    });

    if let Some(event) = event {
        Response::builder()
            .header("content-type", "application/json")
            .status(http::StatusCode::OK)
            .body(event)
            .unwrap()
    } else {
        Response::builder()
            .status(http::StatusCode::NOT_FOUND)
            .body("".to_string())
            .unwrap()
    }
}

async fn replay_events(
    ws: Result<ws::WebSocketUpgrade, ws::rejection::WebSocketUpgradeRejection>,
    State(app): State<Arc<AppState>>,
    Path(game_id): Path<String>,
) -> Response {
    match ws {
        Ok(upgrade) => upgrade.on_upgrade(move |socket| handle_socket(app, game_id, socket)),
        Err(err) => {
            tracing::error!("Failed to upgrade websocket: {}", err);
            http::StatusCode::BAD_REQUEST.into_response()
        }
    }
}

async fn handle_socket(app: Arc<AppState>, game_id: String, mut socket: ws::WebSocket) {
    let events = app.db.get_events(&game_id).unwrap_or_else(|err| {
        tracing::error!("Failed to get events: {}", err);
        Vec::new()
    });

    for event in events {
        println!("{:?}", event);
        if let Err(err) = socket.send(ws::Message::Text(event)).await {
            tracing::error!("Failed to send message: {}", err);
        };
    }

    if let Err(e) = socket.send(ws::Message::Close(None)).await {
        tracing::error!("Could not send Close due to {e}, probably it is ok?");
    }
}

#[derive(serde::Deserialize)]
struct GameSocket {
    engine: String,
    game: String,
}

async fn record_events(
    State(app): State<Arc<AppState>>,
    Query(GameSocket { engine, game }): Query<GameSocket>,
) -> impl IntoResponse {
    let ws_engine = engine
        .replace("http://", "ws://")
        .replace("https://", "wss://");
    let uri = format!("{}/games/{}/events", ws_engine, game);
    let mut ws_stream = match tokio_tungstenite::connect_async(uri).await {
        Ok((stream, _response)) => stream,
        Err(err) => {
            tracing::error!("Failed to connect: {}", err);
            return Response::builder()
                .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                .body("Failed to connect to engine".to_string())
                .unwrap();
        }
    };

    while let Some(msg) = ws_stream.next().await {
        match msg {
            Ok(Message::Text(event)) => {
                println!("Received message: {:?}", event);
                if let Err(err) = app.db.insert_event(&game, &event) {
                    tracing::error!("Failed to write event to database: {}", err);
                    return Response::builder()
                        .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                        .body("Failed to write event to database".to_string())
                        .unwrap();
                }
            }
            Ok(Message::Close(_)) => {
                // Websocket is closing, there's no more messages coming
                // down the pipe so we can just return from this handler
                break;
            }
            Ok(anything_else) => {
                tracing::warn!("Received non-text message: {:?}", anything_else);
            }
            Err(err) => {
                tracing::error!("Error reading message: {}", err);
                return Response::builder()
                    .status(http::StatusCode::INTERNAL_SERVER_ERROR)
                    .body("Error reading message".to_string())
                    .unwrap();
            }
        }
    }

    return Response::builder()
        .status(http::StatusCode::OK)
        .body("OK".into())
        .unwrap();
}

struct AppState {
    agent: sssnake::strategy::Agent,
    db: sssnake::db::Database,
}

async fn agent_server() -> Result<(), Error> {
    let state = std::sync::Arc::new(AppState {
        agent: sssnake::strategy::Agent::default(),
        db: sssnake::db::Database::open()?,
    });

    // Spawn an actual thread to run our train loop
    //let agent_state_clone = agent_state.clone();
    //std::thread::spawn(|| agent_state_clone.train());
    //

    let app = Router::new()
        .route("/", get(bot_info))
        .route("/start", post(initialize_state))
        .route("/move", post(generate_move))
        .route("/end", post(terminate_game))
        .route("/record", get(record_events))
        .route("/games/:game_id", get(game_event_metadata))
        .route("/games/:game_id/events", get(replay_events))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin("https://board.battlesnake.com".parse::<http::HeaderValue>()?)
                .allow_methods(vec![http::Method::GET])
                .allow_headers(vec![http::header::CONTENT_TYPE, http::header::ACCEPT]),
        );

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
        .with_target("sssnake", tracing::Level::INFO)
        .with_target("tower_http", tracing::Level::DEBUG)
        .with_target("axum::rejection", tracing::Level::TRACE);

    tracing_subscriber::registry()
        .with(targets)
        .with(tracing_subscriber::fmt::layer())
        .init();

    agent_server().await?;

    Ok(())
}
