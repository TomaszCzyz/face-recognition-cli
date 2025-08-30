use crate::PROJECT_DIRS;
use crate::person_registry::person_registry::PersonRegistry;
use dlib_wrappers::face_encoding::FaceEncoding;
use sqlite_vec::sqlite3_vec_init;
use sqlx::{Pool, Sqlite, sqlite::SqlitePoolOptions};
use std::fs;
use std::fs::OpenOptions;
use tracing::info;

type Db = Pool<Sqlite>;

pub(crate) struct PersonRegistrySqlite {
    db: Db,
}

impl PersonRegistry for PersonRegistrySqlite {
    /// Gets all encodings with distance lower than a threshold from provided encoding
    fn get(&self, encoding: &FaceEncoding) -> Option<String> {
        let string = "asd".to_string();
        Some(string)
        // todo!()
    }

    fn add(&mut self, encoding: FaceEncoding, name: String) {
        todo!()
    }
}

impl PersonRegistrySqlite {
    pub async fn initialize() -> Self {
        let db = PersonRegistrySqlite::setup_db().await;

        Self { db }
    }

    async fn setup_db() -> Db {
        let mut path = PROJECT_DIRS.data_dir().to_path_buf();

        match fs::create_dir_all(path.clone()) {
            Ok(_) => {}
            Err(err) => {
                panic!("error creating directory {}", err);
            }
        };

        path.push("db.sqlite");

        let result = OpenOptions::new().create(true).write(true).open(&path);

        match result {
            Ok(_) => println!("database file created"),
            Err(err) => panic!("error creating database file {}", err),
        }

        unsafe {
            libsqlite3_sys::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite3_vec_init as *const (),
            )));
        }

        let db = SqlitePoolOptions::new()
            .connect(path.to_str().unwrap())
            .await
            .unwrap();

        info!("Executing migrations...");
        println!("Executing migrations...");
        sqlx::migrate!("./src/migrations").run(&db).await.unwrap();

        let version: (String,) = sqlx::query_as("SELECT sqlite_version();")
            .fetch_one(&db)
            .await
            .unwrap();

        let vec_version: (String,) = sqlx::query_as("SELECT vec_version();")
            .fetch_one(&db)
            .await
            .unwrap();

        println!("sqlite version: {:?}", version);
        println!("vec version: {:?}", vec_version);

        sqlx::query(
            "
            PRAGMA busy_timeout = 60000;
            PRAGMA journal_mode = WAL;
        ",
        )
        .execute(&db)
        .await
        .unwrap();

        db
    }
}
