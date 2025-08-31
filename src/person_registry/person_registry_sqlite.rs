use crate::PROJECT_DIRS;
use crate::person_registry::person_registry::PersonRegistry;
use blake3::Hash;
use dlib_wrappers::Rectangle;
use dlib_wrappers::face_detection::FaceLocations;
use dlib_wrappers::face_encoding::FaceEncoding;
use sqlite_vec::sqlite3_vec_init;
use sqlx::types::chrono::{DateTime, Utc};
use sqlx::{FromRow, Pool, Sqlite, sqlite::SqlitePoolOptions};
use std::fs;
use std::fs::OpenOptions;
use tracing::info;

type Db = Pool<Sqlite>;

pub(crate) struct PersonRegistrySqlite {
    db: Db,
}

#[derive(FromRow)]
pub struct ProcessedFileInsert {
    pub hash: Hash,
    pub path: String,
}

impl PersonRegistrySqlite {
    pub async fn find_file(&self, hash: &Hash) -> Option<(i64, String, DateTime<Utc>)> {
        let maybe_hash: Option<(i64, String, DateTime<Utc>)> =
            sqlx::query_as("SELECT Id, Path, ProcessedAt FROM ProcessedFiles WHERE Hash = $1")
                .bind(&hash.as_bytes()[..])
                .fetch_optional(&self.db)
                .await
                .unwrap();

        maybe_hash
    }

    pub async fn add_file(&self, file: ProcessedFileInsert) -> i64 {
        info!("inserting a new hash for file {}", file.path);
        let res = sqlx::query("INSERT INTO ProcessedFiles (Hash, Path) VALUES ($1, $2)")
            .bind(&file.hash.as_bytes()[..])
            .bind(&file.path)
            .execute(&self.db)
            .await
            .unwrap();

        res.last_insert_rowid()
    }

    /// Gets all encodings with distance lower than a threshold from provided encoding
    pub fn get(&self, encoding: &FaceEncoding) -> Option<String> {
        let string = "asd".to_string();
        Some(string)
        // todo!()
    }

    pub async fn add_face_location(&self, file_id: i64, face_location: &Rectangle) -> i64 {
        let res = sqlx::query(
            "INSERT INTO FaceLocations
                     (Top, \"Left\",Bottom, \"Right\")
                 VALUES
                     ($1, $2, $3, $4)
                 ",
        )
        .bind(face_location.top as i64)
        .bind(face_location.left as i64)
        .bind(face_location.bottom as i64)
        .bind(face_location.right as i64)
        .execute(&self.db)
        .await
        .unwrap();

        let id = res.last_insert_rowid();

        info!("inserted a new face locations: {id} for file: {file_id}");
        id
    }

    pub fn add_encoding(&mut self, encoding: FaceEncoding, hash: Hash) {
        info!("inserting a new encoding");

        // let _ = sqlx::query("INSERT INTO ProcessedFiles (Hash, Path) VALUES ($1, $2)")
        //     .bind(&file.hash.as_bytes()[..])
        //     .bind(&file.path)
        //     .execute(&self.db)
        //     .await
        //     .unwrap();
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
