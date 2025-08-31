use crate::PROJECT_DIRS;
use blake3::Hash;
use dlib_wrappers::Rectangle;
use dlib_wrappers::face_encoding::FaceEncoding;
use sqlite_vec::sqlite3_vec_init;
use sqlx::types::chrono::{DateTime, Utc};
use sqlx::{FromRow, Pool, Sqlite, sqlite::SqlitePoolOptions};
use std::fs;
use std::fs::OpenOptions;
use tracing::info;
use zerocopy::IntoBytes;

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

    pub async fn locate_similar(&self, encoding_id: i64) {
        let res: Vec<(i64, f32)> = sqlx::query_as(
            "
            -- noinspection SqlResolve
            SELECT
              fe.id,
              vec_distance_L2(fe.FaceEncoding, q.vec) AS distance
            FROM FaceEncodings AS fe
            CROSS JOIN (SELECT FaceEncoding AS vec FROM FaceEncodings WHERE id = $1) AS q
            WHERE distance > 0.6
            ORDER BY distance DESC
            LIMIT 30;
            ",
        )
        .bind(&encoding_id)
        .fetch_all(&self.db)
        .await
        .unwrap();

        for (id, dist) in res.iter() {
            println!("{id}: distance: {dist}")
        }
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

    pub async fn add_face_location(&self, face_location: &Rectangle) -> i64 {
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

        info!("inserted a new face locations: {id}");
        id
    }

    pub(crate) async fn add_face_encoding(&self, face_encoding: &FaceEncoding) -> i64 {
        let floats_f32: Vec<f32> = face_encoding.to_vec().iter().map(|&d| d as f32).collect();

        let res = sqlx::query(
            "INSERT INTO FaceEncodings
                     (FaceEncoding)
                 VALUES
                     ($1)
                 ",
        )
        .bind(floats_f32.as_bytes())
        .execute(&self.db)
        .await
        .unwrap();

        let id = res.last_insert_rowid();

        info!("inserted a new face encoding: {id}");
        id
    }

    pub(crate) async fn add_face(&self, file_id: i64, location_id: i64, encoding_id: i64) -> i64 {
        let res = sqlx::query(
            "INSERT INTO Faces
                     (FileId, EncodingId, LocationId)
                 VALUES
                     ($1, $2, $3)
                ",
        )
        .bind(file_id)
        .bind(location_id)
        .bind(encoding_id)
        .execute(&self.db)
        .await
        .unwrap();

        let id = res.last_insert_rowid();

        info!("inserted a new face: {id}");
        id
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
            Ok(_) => {}
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
        sqlx::migrate!("./src/migrations").run(&db).await.unwrap();

        let version: (String,) = sqlx::query_as("SELECT sqlite_version();")
            .fetch_one(&db)
            .await
            .unwrap();

        let vec_version: (String,) = sqlx::query_as(
            "
            -- noinspection SqlResolve
            SELECT vec_version();
            ",
        )
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
