use anyhow::{Context, Error};
use rusqlite::Connection;

const DB_NAME: &str = "sssnake.sqlite";

pub struct Database(Connection);
// Sqlite internally handles locking, so we can safely share the connection
unsafe impl Send for Database {}
unsafe impl Sync for Database {}

fn create_events_table(conn: &Connection) -> Result<(), Error> {
    let _ = conn.execute(
        "CREATE TABLE IF NOT EXISTS events (
            game_id TEXT NOT NULL,
            event TEXT NOT NULL
        ) STRICT",
        [],
    )?;

    Ok(())
}

impl Database {
    pub fn open() -> Result<Self, Error> {
        let home_dir = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .map_err(|err| anyhow::anyhow!("No $HOME variable set: {}", err))?;

        let db_path = home_dir.join(".local").join("state").join(DB_NAME);

        let conn = Connection::open(db_path).context("Failed to open database")?;

        create_events_table(&conn)?;
        Ok(Self(conn))
    }

    pub fn insert_event(&self, game_id: &str, event: &str) -> Result<(), Error> {
        let mut stmt = self
            .0
            .prepare_cached("INSERT INTO events (game_id, event) VALUES (?, ?)")?;
        stmt.execute([game_id, event])?;
        Ok(())
    }

    pub fn get_events(&self, game_id: &str) -> Result<Vec<String>, Error> {
        let mut stmt = self
            .0
            .prepare_cached("SELECT event FROM events WHERE game_id = ?")?;
        let mut rows = stmt.query([game_id])?;

        let mut events = Vec::new();
        while let Some(row) = rows.next()? {
            let event: String = row.get(0)?;
            events.push(event);
        }

        Ok(events)
    }

    pub fn get_game_metadata(&self, game_id: &str) -> Result<Option<String>, Error> {
        let mut stmt = self.0.prepare_cached(
            "SELECT json_object('Game', event->'$.Data') FROM events WHERE game_id = ? AND event->>'$.Type' = 'game_end' LIMIT 1",
        )?;

        let mut rows = stmt.query([game_id])?;
        if let Some(row) = rows.next()? {
            let event: String = row.get(0)?;
            Ok(Some(event))
        } else {
            Ok(None)
        }
    }
}
